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

from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
#from tvm.driver.tvmc import TVMCException

from ...dataflow_pattern import wildcard, is_op, is_constant, is_expr

# don't remove this import even if it does not seem to be used
# because this is the point where the soma_dory backend is registered
import tvm.relay.backend.contrib.soma_dory
from tvm.relay.backend.contrib.soma_dory.transform import SomaDoryGraphQuantizer, SomaDoryLayoutTransform


logger = logging.getLogger("SomaDory")


def _requant_clip_pattern(prev_op):
    """Add pattern requant-optional_clip to prev_op"""

    right_shift = is_op("right_shift")(prev_op, is_constant())
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "int8"})
    # optionally have extra clip/ReLU
    act_or_cast = cast.optional(lambda x: is_op("clip")(x))
    return act_or_cast


def _biasadd_requant_clip_pattern(linear_op):
    """Add pattern bias_add-requant-optional_clip to linear_op"""

    bias_add = is_op("nn.bias_add")(linear_op, wildcard())
    return _requant_clip_pattern(bias_add)


def conv2d_pattern():
    """Create pattern for conv2D with optional fused relu."""

    conv2d = is_op("nn.conv2d")(
            wildcard(), wildcard()
    )
    return _biasadd_requant_clip_pattern(conv2d)


def fully_connected_pattern():
    """Create pattern for nn.dense with optional fused relu."""

    fc = is_op("nn.dense")(
        wildcard(), wildcard()
    )
    return _biasadd_requant_clip_pattern(fc)


def element_wise_add_pattern():
    """Create pattern for element-wise-add with optional fused relu."""

    cast_a = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    cast_b = is_op("cast")(wildcard()).has_attr({"dtype": "int32"})
    add = is_op("add")(cast_a, cast_b)
    return _requant_clip_pattern(add)


def _check_requant_clip(pattern):
    """Check if requant-clip pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the op before this sequence if supported
    """
    if str(pattern.op.name) == "clip":
        clip = pattern
        cast = clip.args[0]
    else:
        cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning("shift factor of accelerator operation must be in range [0, 31], but got {shift_factor}. Acceleration for this op is not supported")
        return None

    right_shift_input = right_shift.args[0]

    return right_shift_input


def _check_biasadd_requant_clip(pattern):
    """Check if bias_add-requant-clip pattern is supported by the soma dory accelerator
    Returns None if not supported, returns the linear op before this sequence if supported
    """

    right_shift_input = _check_requant_clip(pattern)
    if right_shift_input is None:
        return None

    # For now, we don't support linears without bias
    if str(right_shift_input.op.name) != "nn.bias_add":
        logger.warning("Found conv/dense op without nn.bias_add. Acceleration for this op is not supported")
        return None

    bias_add = right_shift_input

    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this op is not supported")
        return None

    return bias_add.args[0]


def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""

    conv2d = _check_biasadd_requant_clip(pattern)
    if conv2d is None:
        return False

    num_output_channels = conv2d.args[1].data.shape[0]

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected nn.conv2d {name} to be one of {supported_values}, but got {attr}. \
                            Acceleration for this conv2d is not supported")
            return False

        return True

    # check conv2d attributes
    if (#not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_size', [[1, 1], [3, 3], [5, 5], [7, 7]])
        #or not is_conv2d_attr_value_supported(conv2d.attrs, 'padding', [4*[0], 4*[1], [1, 1, 0, 0], [0, 0, 1, 1]])
        not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1, num_output_channels])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

        return False

    #conv2d_input = conv2d.args[0]
    #conv2d_weight = conv2d.args[1]

    return True


def check_fully_connected(pattern):
    """Check if the fully connected layer is supported by the soma dory accelerator"""

    fc = _check_biasadd_requant_clip(pattern)
    if fc is None:
        return False

    #fc_input = fc.args[0]
    #fc_weight = fc.args[1]

    return True


def check_element_wise_add(pattern):
    """Check if the element-wise-add layer is supported by the soma dory accelerator"""

    add = _check_requant_clip(pattern)
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
        ("soma_dory.conv2d", conv2d_pattern(), check_conv2d),
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

    pipeline = [
            SomaDoryGraphQuantizer('int8'),
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget(["soma_dory"]),
            transform.InferType(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]

    if 'layout_transform' not in opts or opts['layout_transform'] != '0':
        pipeline.insert(4, SomaDoryLayoutTransform())

    seq = tvm.transform.Sequential(pipeline)

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
