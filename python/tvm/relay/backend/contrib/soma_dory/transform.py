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

