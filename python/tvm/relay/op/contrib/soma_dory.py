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


def conv2d_pattern():
    """Create pattern for conv2D with optional fused relu."""
    conv2d = is_op("nn.conv2d")(
        wildcard(), wildcard()
    )
    bias_add = is_op("nn.bias_add")(conv2d, wildcard())
    right_shift = is_op("right_shift")(bias_add,
                                       is_constant())
    # TODO: figure out how to match on attributes for clip?
    clip = is_op("clip")(right_shift)
    cast = is_op("cast")(clip).has_attr({"dtype": "int8"})
    # optionally have extra clip/ReLU
    act_or_cast = cast.optional(lambda x: is_op("clip")(x))
    return act_or_cast


def check_conv2d(pattern):
    """Check if the Conv2D is supported by the soma dory accelerator"""

    if str(pattern.op.name) == "clip":
        clip = pattern
        cast = clip.args[0]
    else:
        cast = pattern
    right_shift = cast.args[0].args[0]

    # Check range of shift factor
    shift_factor = right_shift.args[1].data.numpy()
    if shift_factor < 0 or shift_factor > 31:
        logger.warning("Conv2d shift factor must be in range [0, 31], but got {shift_factor}. Acceleration for this conv2d is not supported")
        return False

    right_shift_input = right_shift.args[0]

    # For now, we don't support convolutions without bias
    if str(right_shift_input.op.name) != "nn.bias_add":
        logger.warning("Found convolution without nn.bias_add. Acceleration for this conv2d is not supported")
        return False

    bias_add = right_shift_input

    # Check bias dtype
    bias_dtype = bias_add.args[1].checked_type.dtype
    if bias_dtype != 'int32':
        logger.warning(f"Expected nn.bias_add parameters to be of type int32, but got {bias_dtype}. Acceleration for this conv2d is not supported")
        return False

    conv2d = bias_add.args[0]

    def is_conv2d_attr_value_supported(attrs, name, supported_values):
        attr = attrs[name]

        if isinstance(attr, tvm.ir.container.Array):
            attr = list(attr)

        if attr not in supported_values:
            logger.warning(f"Expected qnn.conv2d {name} to be one of {supported_values}, but got {attr}. \
                            Acceleration for this conv2d is not supported")
            return False

        return True

    # check conv2d attributes
    if (not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_size', [[1, 1], [3, 3], [5, 5], [7, 7]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'padding', [4*[0], 4*[1], [1, 1, 0, 0], [0, 0, 1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'strides', [[1, 1], [2, 2]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'dilation', [[1, 1]])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'groups', [1])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'kernel_layout', ['OIHW'])
        or not is_conv2d_attr_value_supported(conv2d.attrs, 'data_layout', ['NCHW'])):

        return False

    #conv2d_input = conv2d.args[0]
    #conv2d_weight = conv2d.args[1]

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

    seq = tvm.transform.Sequential(
        [
            SomaDoryGraphQuantizer('int8'),
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget(["soma_dory"]),
            SomaDoryLayoutTransform(),
            transform.InferType(),
            transform.PartitionGraph(),
            transform.InferType(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        try:
            return seq(mod)
        except Exception as err:
            raise Exception(
                "Error converting layout to {0}".format(str(err))
            )
