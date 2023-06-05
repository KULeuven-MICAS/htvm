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
"""Relay transformations for Diana ONNX"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import DFPatternCallback, rewrite, wildcard, is_op, is_constant
from functools import partial


class DianaOnnxIntegerizeLinearOps(DFPatternCallback):
    """For earch linear op, cast weights to int8, set linear op output to int32 and cast optional bias to int32.
    """
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        self.w = is_constant()
        self.b = is_constant()

        linear = is_op("nn.conv2d")(self.x, self.w) | is_op("nn.dense")(self.x, self.w)
        # NOTE: onnx parser of TVM produces 'add' rather than 'bias_add' for Gemm so we match on 'add' too
        self.pattern = linear.optional(lambda x: is_op("nn.bias_add")(x, self.b) | is_op("add")(x, self.b))

    def callback(self, pre, post, node_map):
        call = pre

        # construct integerized bias call
        new_bias_add_call = None
        if 'add' in call.op.name:
            call = call.args[0]
            b = node_map[self.b][0]
            b = relay.const(b.data.numpy().astype('int32'))
            new_bias_add_call = partial(relay.op.nn.bias_add, bias=b)

        # construct integerized linear call
        w = node_map[self.w][0]
        w = relay.const(w.data.numpy().astype('int8'))

        if call.op.name == 'nn.conv2d':
            new_linear_call = partial(relay.op.nn.conv2d,
                                      weight=w,
                                      strides=call.attrs.strides,
                                      padding=call.attrs.padding,
                                      dilation=call.attrs.dilation,
                                      groups=call.attrs.groups,
                                      out_dtype='int32',
                                      kernel_size=w.data.shape[-2:])
        else:
            new_linear_call = partial(relay.op.nn.dense,
                                      weight=w,
                                      out_dtype='int32')

        # write new subgraph
        x = node_map[self.x][0]
        x = new_linear_call(x)
        if new_bias_add_call is not None:
            x = new_bias_add_call(x)

        return x


class DianaOnnxIntegerizeElementWiseSum(DFPatternCallback):
    """For earch element-wise sum, add cast to int32 at both inputs.
    """
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        # NOTE: has_attr({}) is a trick to not match to constants to avoid matching to bias-add-like add ops
        self.a = wildcard().has_attr({})
        self.b = wildcard().has_attr({})

        self.pattern = is_op("add")(self.a, self.b)

    def callback(self, pre, post, node_map):
        a = node_map[self.a][0]
        b = node_map[self.b][0]

        a = relay.op.cast(a, 'int32')
        b = relay.op.cast(b, 'int32')
        y = relay.op.add(a, b)

        return y


class DianaOnnxMergeDuplicateDiv(DFPatternCallback):
    """Merge consecutive div ops with constants together (workaround for bug in quantlib)
    """
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        self.div1 = is_constant()
        self.div2 = is_constant()

        div1 = is_op("divide")(self.x, self.div1)
        self.pattern = is_op("divide")(div1, self.div2)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div1 = node_map[self.div1][0]
        div2 = node_map[self.div2][0]

        div = div1.data.numpy() * div2.data.numpy()
        return relay.op.divide(x, relay.const(div))


def create_requant(x, div_factor, a_min, a_max, shift_factor_dtype='int32'):
    """Create either:

         1)    right_shift -> clip -> cast(int8)
         2)    div -> clip -> cast(int8)

       on top of 'x', based on division factor div_factor
       Create pattern 1 in case x is not an input (Var) and div_factor is a power-of-two value > 1
       Create pattern 2 otherwise

    """
    div_factor_log2 = np.log2(div_factor)
    x_is_float_var = isinstance(x, relay.Var) and x.type_annotation.dtype == 'float32'

    # check if division can be replaced by a right_shift
    if not x_is_float_var and \
       div_factor_log2 == np.round(div_factor_log2) and \
       div_factor_log2 >= 0:

        shift_factor = div_factor_log2.astype(shift_factor_dtype)
        assert shift_factor.size == 1
        x = relay.op.right_shift(x, relay.const(shift_factor[0]))
    else:
        x = relay.op.divide(x, relay.const(div_factor))

    x = relay.op.clip(x, a_min=a_min, a_max=a_max)
    x = relay.op.cast(x, 'int8')

    return x


class DianaOnnxDigitalRequantRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    Rewrite: div -> floor -> max -> min
         to: right_shift -> clip -> cast(int8)
         or: div -> clip -> cast(int8)
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.x = wildcard()
        self.div = is_constant()
        self.maximum = is_constant()
        self.minimum = is_constant()

        div = is_op("divide")(self.x, self.div)
        floor = is_op("floor")(div)
        maximum = is_op("maximum")(floor, self.maximum)
        self.pattern = is_op("minimum")(maximum, self.minimum)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div = node_map[self.div][0]
        maximum = node_map[self.maximum][0]
        minimum = node_map[self.minimum][0]

        return create_requant(x, div.data.numpy(),
                              int(maximum.data.numpy()),
                              int(minimum.data.numpy()))


class DianaOnnxAnalogBnRequantRewriter(DFPatternCallback):
    """Rewriter for analog batchnorm + requant pattern
    Rewrite: cast -> div -> floor -> max -> min -> mul -> add -> div -> floor -> max -> min (-> cast)
         to: cast -> div -> floor -> clip -> cast -> mul -> add -> right_shift -> clip -> cast
    """

    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.x = wildcard()
        self.div1 = is_constant()
        self.maximum1 = is_constant()
        self.minimum1 = is_constant()
        self.mul = is_constant()
        self.add = is_constant()
        self.div2 = is_constant()
        self.maximum2 = is_constant()
        self.minimum2 = is_constant()

        div1 = is_op("divide")(self.x, self.div1)
        floor1 = is_op("floor")(div1)
        maximum1 = is_op("maximum")(floor1, self.maximum1)
        minimum1 = is_op("minimum")(maximum1, self.minimum1)
        mul = is_op("multiply")(minimum1, self.mul)
        add = is_op("add")(mul, self.add)
        div2 = is_op("divide")(add, self.div2)
        floor2 = is_op("floor")(div2)
        maximum2 = is_op("maximum")(floor2, self.maximum2)
        self.pattern = is_op("minimum")(maximum2, self.minimum2)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div1 = node_map[self.div1][0]
        maximum1 = node_map[self.maximum1][0]
        minimum1 = node_map[self.minimum1][0]
        mul = node_map[self.mul][0]
        add = node_map[self.add][0]
        div2 = node_map[self.div2][0]
        maximum2 = node_map[self.maximum2][0]
        minimum2 = node_map[self.minimum2][0]

        x = relay.op.cast(x, 'float')
        x = relay.op.divide(x, div1)
        x = relay.op.floor(x)
        x = relay.op.clip(x, a_min=int(maximum1.data.numpy()), a_max=int(minimum1.data.numpy()))
        x = relay.op.cast(x, 'int16')
        x = relay.op.multiply(x, relay.const(mul.data.numpy().astype('int16')))
        x = relay.op.add(x, relay.const(add.data.numpy().astype('int16')))

        return create_requant(x, div2.data.numpy(),
                              int(maximum2.data.numpy()),
                              int(minimum2.data.numpy()), 'int16')


class DianaOnnxCorrectDequant(DFPatternCallback):
    """Rewrite each dequant op from:
            cast(int8) -> div
        to:
            cast(int8) -> cast(float) -> div
        this is needed since the output of the dequant is assumed to be float
    """
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        self.div = is_constant()

        cast = is_op("cast")(self.x).has_attr({'dtype': 'int8'})
        self.pattern = is_op("divide")(cast, self.div)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div = node_map[self.div][0]

        x = relay.op.cast(x, 'int8')
        x = relay.op.cast(x, 'float')
        x = relay.op.divide(x, div)

        return x


@tvm.ir.transform.module_pass(opt_level=0)
class DianaOnnxIntegerize:
    """ Transform relay graph for DIANA, parsed from quantlib ONNX, to integer operations.
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(DianaOnnxIntegerizeLinearOps(), func)
            func = rewrite(DianaOnnxIntegerizeElementWiseSum(), func)
            func = rewrite(DianaOnnxMergeDuplicateDiv(), func)
            func = rewrite(DianaOnnxAnalogBnRequantRewriter(), func)
            func = rewrite(DianaOnnxDigitalRequantRewriter(), func)
            func = rewrite(DianaOnnxCorrectDequant(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
