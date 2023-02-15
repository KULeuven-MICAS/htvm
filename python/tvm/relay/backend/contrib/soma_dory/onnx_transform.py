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
    """For earch linear op, cast weights to int8, optional bias to int32 and insert cast ops before
    and after the linear op.
    """
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        self.w = is_constant()
        self.b = is_constant()

        linear = is_op("nn.conv2d")(self.x, self.w) | is_op("nn.dense")(self.x, self.w)
        self.pattern = linear.optional(lambda x: is_op("nn.bias_add")(x, self.b))

    def callback(self, pre, post, node_map):
        call = pre

        # construct integerized bias call
        new_bias_add_call = None
        if call.op.name == 'nn.bias_add':
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
        x = relay.cast(x, 'int8')
        x = new_linear_call(x)
        if new_bias_add_call is not None:
            x = new_bias_add_call(x)
        x = relay.cast(x, 'float')

        return x


class DianaOnnxDigitalRequantRewriter(DFPatternCallback):
    """Rewriter for digital requant pattern
    Rewrite: cast -> div -> floor -> max -> min (-> cast)
         to: right_shift -> clip -> cast
    """
    def __init__(self, require_type=False):
        super().__init__(require_type)

        self.x = wildcard()
        self.div = is_constant()
        self.maximum = is_constant()
        self.minimum = is_constant()

        cast = is_op("cast")(self.x)
        div = is_op("divide")(cast, self.div)
        floor = is_op("floor")(div)
        maximum = is_op("maximum")(floor, self.maximum)
        minimum = is_op("minimum")(maximum, self.minimum)
        self.pattern = minimum.optional(lambda x: is_op("cast")(x))

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        div = node_map[self.div][0]
        maximum = node_map[self.maximum][0]
        minimum = node_map[self.minimum][0]

        shift_factor = np.log2(div.data.numpy()).astype('int32')
        x = relay.op.right_shift(x, relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(maximum.data.numpy()), a_max=int(minimum.data.numpy()))
        x = relay.op.cast(x, 'int8')

        if pre.op.name != 'cast':
            x = relay.op.cast(x, 'float')

        return x


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

        cast = is_op("cast")(self.x)
        div1 = is_op("divide")(cast, self.div1)
        floor1 = is_op("floor")(div1)
        maximum1 = is_op("maximum")(floor1, self.maximum1)
        minimum1 = is_op("minimum")(maximum1, self.minimum1)
        mul = is_op("multiply")(minimum1, self.mul)
        add = is_op("add")(mul, self.add)
        div2 = is_op("divide")(add, self.div2)
        floor2 = is_op("floor")(div2)
        maximum2 = is_op("maximum")(floor2, self.maximum2)
        minimum2 = is_op("minimum")(maximum2, self.minimum2)
        self.pattern = minimum2.optional(lambda x: is_op("cast")(x))

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

        shift_factor = np.log2(div2.data.numpy()).astype('int16')
        x = relay.op.right_shift(x, relay.const(shift_factor))
        x = relay.op.clip(x, a_min=int(maximum2.data.numpy()), a_max=int(minimum2.data.numpy()))
        x = relay.op.cast(x, 'int8')

        if pre.op.name != 'cast':
            x = relay.op.cast(x, 'float')

        return x


@tvm.ir.transform.module_pass(opt_level=0)
class DianaOnnxIntegerize:
    """ Transform relay graph for DIANA, parsed from ONNX, to efficient integer opterations
        1) Cast weights and biases to integers and insert cast operations before and after linear ops
        2) Rewrite digital requant operations
        3) Rewrite analog requant + batchnorm operations
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(DianaOnnxIntegerizeLinearOps(), func)
            func = rewrite(DianaOnnxAnalogBnRequantRewriter(), func)
            func = rewrite(DianaOnnxDigitalRequantRewriter(), func)
            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
