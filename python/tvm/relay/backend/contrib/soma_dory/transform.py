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
"""Relay transformations for DORY SOMA"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor


@transform.function_pass(opt_level=0)
class SomaDoryGraphQuantizer(ExprMutator):
    """Convert fake-quantized relay graph (from Soma ONNX file) into a real quantized relay graph
    """

    def __init__(self, dtype):
        self.dtype = dtype
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]

        if call.op.name == 'nn.conv2d':
            # cast the weights and output of the conv2d to integers if they are floats (assume they are constant)
            w = new_args[1]
            if not w.data.dtype.startswith('int'):
                w = relay.const(w.data.numpy().astype(self.dtype))

            new_call = relay.op.nn.conv2d(new_args[0], w,
                                          strides=call.attrs.strides,
                                          padding=call.attrs.padding,
                                          dilation=call.attrs.dilation,
                                          groups=call.attrs.groups,
                                          out_dtype='int32',
                                          kernel_size=w.data.shape[-2:])

        elif call.op.name == 'nn.bias_add':
            # cast bias to int32
            new_args[1] = relay.const(new_args[1].data.numpy().astype('int32'))
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        elif call.op.name == 'divide':
            # We currently assume that a divide op represents a requant operations after bias_add or element-wise sum
            # Since the currently existing 'qnn.op.requantize' does not support floor-based rounding, we construct our
            # own requantization using a set of primitive relay ops. We expect that the division factor is power-of-two
            # and therefore our custom requantization is a sequence of these ops: right_shift, clip, cast.
            shift_factor = int(np.log2(new_args[1].data.numpy()))
            right_shift = relay.op.right_shift(new_args[0], relay.const(shift_factor))
            clip = relay.op.clip(right_shift, a_min=-128, a_max=127)
            new_call = relay.op.cast(clip, self.dtype)

        else:
            new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        return new_call

    def visit_function(self, fn):
        """Rewrite function arguments
        """
        new_params = []
        binds = {}

        for param in fn.params:
            # Get the parameter's type annotation.
            var_type = param.type_annotation

            # bias params are int32
            if param.name_hint.endswith('bias'):
                dtype = 'int32'
            else:
                dtype = self.dtype

            # Generate new variable.
            new_param = relay.var(param.name_hint, shape=var_type.shape, dtype=dtype)

            new_params.append(new_param)
            binds[param] = new_param

        new_body = self.visit(fn.body)
        # Rewrite the body to use new parameters.
        new_body = relay.bind(new_body, binds)

        # Construct the updated function and return.
        return relay.Function(
            new_params,
            new_body,
            # You could change the return type, if you use None it will re-infer.
            None,
            type_params=fn.type_params,
            attrs=fn.attrs,
        )


class FindLayoutTransformShape(ExprVisitor):
    """Convert relay graph to dory graph
    """
    def __init__(self):
        super().__init__()
        self.shapes = []

    def visit_call(self, call):
        """Extract parameters and construct dory graph"""
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant):
            # we don't want to insert transformations on constants like weights and biases
            if call.op.name == 'annotation.compiler_begin' and call.attrs.compiler == 'soma_dory':
                self.shapes.append(call.args[0].checked_type.shape)

            elif call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'soma_dory':
                self.shapes.append(call.args[0].checked_type.shape)


@transform.function_pass(opt_level=0)
class SomaDoryLayoutTransform(ExprMutator):
    """Insert soma_dory specific layout transform before and after each 'soma_dory' annotated relay Function
    TODO: make this smart to avoid unnecessary transformations
    """

    def transform_function(self, func, mod, ctx):
        self.f = FindLayoutTransformShape()
        self.f.visit(func)

        return self.visit(func)

    def create_transform(self, x, shape):
        """Create soma_dory layout transform from 'reshape -> reverse -> reshape' op sequence
        """

        x = relay.reshape(x, (np.prod(shape) // 4, 4))
        x = relay.reverse(x, axis=1)
        x = relay.reshape(x, shape)

        return x

    def visit_call(self, call):
        """Rewrite ops
        """
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        new_call = relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

        if isinstance(call.op, tvm.ir.Op) and not isinstance(call.args[0], relay.Constant):
            # we don't want to insert transformations on constants like weights and biases
            if call.op.name == 'annotation.compiler_begin' and call.attrs.compiler == 'soma_dory':
                # insert transformation before this op
                shape = self.f.shapes.pop(0)
                x = self.create_transform(new_args[0], shape)
                new_call = relay.op.annotation.compiler_begin(x, 'soma_dory')

            elif call.op.name == 'annotation.compiler_end' and call.attrs.compiler == 'soma_dory':
                # insert transformation after this op
                shape = self.f.shapes.pop(0)
                new_call = self.create_transform(new_call, shape)

        return new_call

