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
"""Data layout transformations for soma dory"""

import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.dataflow_pattern import DFPatternCallback, FunctionPattern, dominates, rewrite, wildcard, is_op, is_constant


def create_layout_transform(x, shape):
    """Create soma_dory layout transform from 'reshape -> reverse -> reshape' op sequence
    """
    x = relay.reshape(x, (np.prod(shape) // 4, 4))
    x = relay.reverse(x, axis=1)
    x = relay.reshape(x, shape)

    return x


def create_layout_transform_pattern(x=wildcard()):
    """Return a search pattern for matching to a layout transform
    """
    x = is_op('reshape')(x)
    x = is_op('reverse')(x)
    x = is_op('reshape')(x)

    return x


def clone_sub_graph(start, end, target):
    """Clone the subgraph between node 'start' and 'end' ontop of 'target'
    """

    class CloneSubGraph(ExprVisitor):

        def __init__(self, start, target):
            super().__init__()
            self.clone = False
            self.start = start
            self.target = target

        def visit_call(self, call):
            self.visit(call.op)
            for a in call.args:
                self.visit(a)

            if self.clone:
                if len(call.args) == 1:
                    new_args = [self.target]
                else:
                    new_args = [self.target] + call.args[1:]
                self.target = relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)

            if call == self.start:
                self.clone = True

    c = CloneSubGraph(start, target)
    c.visit(end)

    return c.target


class SomaDoryInsertTransformsAroundFunctions(DFPatternCallback):
    """Insert transforms before and after each soma_dory annotated function, except for soma_dory.add
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        # We match each composite function call and decide which call needs a transform in the callback,
        # since we are not able to express this decision with pattern matching.
        self.pattern = FunctionPattern(None, wildcard())(None)

    def callback(self, pre, post, node_map):
        func = pre.op

        # We don't add transforms around soma_dory element-wise add ops
        if func.attrs.Composite == 'soma_dory.add':
            return func(*post.args)

        shape_begin = pre.args[0].checked_type.shape
        shape_end = pre.checked_type.shape

        x = post.args[0]
        x = create_layout_transform(x, shape_begin)
        x = func(x, *post.args[1:])
        x = create_layout_transform(x, shape_end)

        return x


class SomaDoryInsertTransformsDiamondPattern(DFPatternCallback):
    """Insert transformations on diamond patterns
       Example:

           before          after

             |               |
             |              tfm
             |               |
            / \             / \
           /   \           /   \
          |     |        tfm   tfm
          *     *   -->   *     *
          |     |        tfm   tfm
           \   /           \   /
         reduction       reduction
             |               |
             |              tfm
             |               |

       where '*' is any given subgraph, 'reduction' is an element-wise reduction pattern
       and 'tfm' is a layout transform.
    """
    def __init__(self, require_type=True, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        # we match to any ops on both branches and filter out the rest in the callback
        self.a = wildcard()
        self.b = wildcard()
        #reduction = FunctionPattern([wildcard(), wildcard()], wildcard())(self.a, self.b) | \
        reduction = is_op('add')(self.a, self.b)
        fuzzy = FunctionPattern([wildcard(), wildcard(), wildcard()], wildcard())(wildcard(), wildcard(), wildcard())

        self.pattern = dominates(self.x, fuzzy, reduction)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        a = node_map[self.a][0]
        b = node_map[self.b][0]

        shape_input = x.checked_type.shape
        shape_output = a.checked_type.shape

        # rewrite branch with additional transformations
        x_ = create_layout_transform(x, shape_input)
        a_ = create_layout_transform(x_, shape_input)
        b_ = create_layout_transform(x_, shape_input)

        # repeat whatever is on the left (a) side on top of a_
        a_ = clone_sub_graph(x, a, a_)

        # repeat whatever is on the right (b) side on top of b_
        b_ = clone_sub_graph(x, b, b_)

        # rewrite reduction with additional transforms
        a_ = create_layout_transform(a_, shape_output)
        b_ = create_layout_transform(b_, shape_output)
        x = pre.op(a_, b_)
        x = create_layout_transform(x, shape_output)

        return x


class SomaDoryRemoveDuplicateTransforms(DFPatternCallback):
    """Remove consecutive duplicate transforms:
    If we encounter:
        -> layout_transform -> layout_transform ->
    we remove both
    """
    def __init__(self, require_type=False, rewrite_once=True):
        super().__init__(require_type, rewrite_once)

        self.x = wildcard()
        x = create_layout_transform_pattern(self.x)
        self.pattern = create_layout_transform_pattern(x)

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        return x


@tvm.ir.transform.module_pass(opt_level=0)
class SomaDoryLayoutTransform:
    """ Insert diana accelerator data layout transformation where needed. This is done in two steps:
    1) Find all annotated soma_dory functions (except for element-wise add since this op does not need the transformations)
       and insert a 'reshape -> reverse -> reshape' transformation before and after the function
    2) Remove duplicate/unnessasary transformations in a few separate passes
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            func = rewrite(SomaDoryInsertTransformsAroundFunctions(), func)
            print("After insert transform")
            print(func)
            func = rewrite(SomaDoryInsertTransformsDiamondPattern(), func)
            print("After insert transform diamond pattern")
            print(func)
            func = rewrite(SomaDoryRemoveDuplicateTransforms(), func)
            print("After removing transforms")
            print(func)
            mod.update_func(global_var, func)
        mod = transform.InferType()(mod)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
