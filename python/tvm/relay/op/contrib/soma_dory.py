# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Operations to support the SOMA accelerator.
"""

import tvm
import logging
from functools import partial

from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr

# don't remove this import even if it does not seem to be used
# because this is the point where the soma_dory backend is registered
import tvm.relay.backend.contrib.soma_dory
from tvm.relay.backend.contrib.soma_dory.transform import SomaDoryLayoutTransform


logger = logging.getLogger("SomaDory")


def _requant_pattern(prev_op):
    """Add requant pattern (right_shift -> clip -> cast) to prev_op"""
    right_shift = is_op("right_shift")(prev_op, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "int8"})
    return cast


def _biasadd_requant_pattern(linear_op):
    """Add pattern bias_add-requant to linear_op"""

    bias_add = is_op("nn.bias_add")(linear_op, wildcard()) | is_op("add")(linear_op, wildcard())
    return _requant_pattern(bias_add)


def _analog_bn_requant_pattern(prev_op):
    """Add analog batchnorm and requant pattern
       (cast -> div -> floor -> clip -> cast -> mul -> add -> right_shift -> clip -> cast)
    """
    cast1 = is_op("cast")(prev_op).has_attr({"dtype": "float32"})
    div = is_op("divide")(cast1, is_constant())
    floor = is_op("floor")(div)
    clip = is_op("clip")(floor)
    cast2 = is_op("cast")(clip).has_attr({"dtype": "int16"})
    mul = is_op("multiply")(cast2, is_constant())
    add = is_op("add")(mul, is_constant())
    return _requant_pattern(add)


def conv2d_pattern(is_analog):
    """Create pattern for conv2D with optional bias and requantization"""

    conv2d = is_op("nn.conv2d")(wildcard(), wildcard())

    if is_analog:
        return _analog_bn_requant_pattern(conv2d)

    return _biasadd_requant_pattern(conv2d)


def fully_connected_pattern():
    """Create pattern for nn.dense with optional fused relu."""

    fc = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    return _biasadd_requant_pattern(fc)


def element_wise_add_pattern():
    """Create pattern for element-wise-add with optional fused relu."""

    cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    add = is_op("add")(cast_a, cast_b)
    return _requant_pattern(add)


def _check_requant(pattern):
    """Check if requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the op before this sequence, if supported
    """
    cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning(f"shift factor of accelerator operation must be in range [0, 31], but got {shift_factor}. Acceleration for this op is not supported.")
        return None

    right_shift_input = right_shift.args[0]

    return right_shift_input


def _check_analog_bn_requant(pattern):
    """Check if analog batchnorm requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the op before this sequence, if supported
    """
    right_shift_input = _check_requant(pattern)
    if right_shift_input is None:
        return None

    add = right_shift_input
    # Check add ?

    mul = add.args[0]
    # Check mul ?

    clip = mul.args[0].args[0]
    clip_range = [clip.attrs.a_min, clip.attrs.a_max]
    expected_clip_range = [-32, 31]
    if clip_range != expected_clip_range:
        logger.warning(f"Clip range of analog ADC is wrong: expected {expected_clip_range}, but got {clip_range}. Acceleration for this op is not supported.")
        return None

    div = clip.args[0].args[0]
    # Check analog gain factor ?

    return div.args[0].args[0]


def _check_biasadd_requant(pattern):
    """Check if bias_add-requant pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the linear op before this sequence, if supported
    """

    right_shift_input = _check_requant(pattern)
    if right_shift_input is None:
        return None

    bias_add = right_shift_input

    # We can safely assumer bias is present since pattern matcher expects this
    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this op is not supported.")
        return None

    return bias_add.args[0]

def _check_conv2d(conv2d, is_analog):
    """Check Conv2d attributes requirements"""

    core_type_str = "analog" if is_analog else "digital"

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected {core_type_str} nn.conv2d {name} to be one of {supported_values}, but got {attr}. " +\
                            "Acceleration for this op is not supported.")
            return False

        return True

    def is_filter_and_padding_supported(attrs):
        kernel_size = list(attrs["kernel_size"])
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]
        supported_kernels = [1, 3, 5, 7]
        if (kernel_h not in supported_kernels) or (kernel_w not in supported_kernels):
            logger.warning(f"Expected {core_type_str} nn.conv2d kernel width and height to be one of {supported_kernels}, " +\
                           f"but got {kernel_size}. " +\
                            "Acceleration for this op is not supported.")
            return False

        # In topi, padding is [padt, padl, padb, padr]
        padding = list(attrs["padding"])
        # Only support equal left-right and top-bottom padding
        if (padding[0] != padding[2]) or (padding[1] != padding[3]):
            logger.warning(f"Expected {core_type_str} nn.conv2d to have equal top and bottom padding, and equal left and right padding," +\
                           f"but got {[padding[0], padding[2]]} and {[padding[1], padding[3]]}, respectively. " +\
                            "Acceleration for this op is not supported.")
            return False

        # Only support output with same output dimension on accelerator
        if (kernel_w - 2*padding[1] != 1) and (kernel_h - 2*padding[0] != 1):
            expected_pad = [(kernel_w - 1) // 2, (kernel_h - 1) // 2]
            logger.warning(f"Accelerator only supports 'SAME' padding. " +\
                           f"Expected {core_type_str} nn.conv2d with kernel size {kernel_size} to have padding {expected_pad}, " +\
                           f"but got {padding[:2]}.")
            return False

        return True


    # check conv2d attributes
    num_output_channels = conv2d.args[1].data.shape[0]
    supported_groups = [1] if is_analog else [1, num_output_channels]
    if (not is_filter_and_padding_supported(conv2d.attrs)
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', supported_groups)
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

        return False

    conv2d_weight = conv2d.args[1]
    weights_dtype = conv2d_weight.data.dtype
    if not weights_dtype == 'int8':
        logger.warning(f"Expected {core_type_str} Conv2D weights data type to be int8, got {weights_dtype}. " +\
                        "Acceleration for this conv2d is not supported")
        return False

    return True


def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""

    conv2d = _check_biasadd_requant(pattern)
    if conv2d is None:
        return False

    if not _check_conv2d(conv2d, False):
        return False

    return True

def check_analog_conv2d(pattern):
    """Check if the analog Conv2D is supported by the soma dory accelerator"""

    conv2d = _check_analog_bn_requant(pattern)
    if conv2d is None:
        return False

    if not _check_conv2d(conv2d, True):
        return False

    # Verify max and min weight values
    w = conv2d.args[1].data.numpy()
    if w.min() < -1 or w.max() > 1:
        logger.warning(f"Expected analog Conv2D weights to be in range [-1, 1], but got {[w.min(), w.max()]}. " +\
                        "Acceleration for this conv2d is not supported")
        return False


    return True


def check_fully_connected(pattern):
    """Check if the fully connected layer is supported by the soma dory accelerator"""

    fc = _check_biasadd_requant(pattern)
    if fc is None:
        return False

    #fc_input = fc.args[0]
    #fc_weight = fc.args[1]

    return True


def check_element_wise_add(pattern):
    """Check if the element-wise-add layer is supported by the soma dory accelerator"""
    add = _check_requant(pattern)
    if add is None:
        return False

    tensor_shape_a = list(add.args[0].checked_type.shape)
    tensor_shape_b = list(add.args[1].checked_type.shape)
    if tensor_shape_a != tensor_shape_b:
        logger.warning(f"Tensor shapes for element-wise-add don't match:"+\
                " Tensor a: {tensor_shape_a}," + \
                " Tensor b: {tensor_shape_b}." + \
                " Acceleration for this element-wise-add is not supported")
        return False

    return True


def pattern_table():
    """
    Registers the patterns we want to match.
    Returns
    -------
        The patterns.
    """

    return [
        ("soma_dory.aconv2d", conv2d_pattern(True), check_analog_conv2d),
        ("soma_dory.conv2d", conv2d_pattern(False), check_conv2d),
        ("soma_dory.dense", fully_connected_pattern(), check_fully_connected),
        ("soma_dory.add", element_wise_add_pattern(), check_element_wise_add),
    ]


def partition_for_soma_dory(mod, params=None, dpu=None, **opts):
    """
    The partitioning sequence for the soma_dory byoc
    Parameters
    ----------
    mod The module to use

    Returns
    -------
    The partitioned module.

    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    pipeline = []

    pipeline.append(tvm.transform.PrintIR())
    pipeline.append(transform.MergeComposite(pattern_table()))
    pipeline.append(transform.AnnotateTarget(["soma_dory"]))

    if 'layout_transform' not in opts or opts['layout_transform'] != '0':
        pipeline.append(SomaDoryLayoutTransform())

    pipeline.append(transform.InferType())
    pipeline.append(transform.PartitionGraph())
    pipeline.append(transform.InferType())
    pipeline.append(tvm.transform.PrintIR())

    seq = tvm.transform.Sequential(pipeline)

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
