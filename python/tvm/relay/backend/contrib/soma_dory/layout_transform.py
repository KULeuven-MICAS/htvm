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

import tvm
import numpy as np
import networkx as nx
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


def calculate_transforms(g):
    """Calculate where layout transforms need to be enabled in the networkx graph.
    We start with a networkx graph with all layout transforms disabled (tfm = False)
    """

    def all_edges(g, node):
        return list(g.in_edges(node)) + list(g.out_edges(node))

    def toggle_tfms(g, edges):
        for edge in edges:
            g.edges[edge]['tfm'] = not g.edges[edge]['tfm']

    # enable transforms around 'acc' nodes
    for node, data in g.nodes(data=True):
        if data['layout'] == 'acc':
            toggle_tfms(g, all_edges(g, node))

    # optimize transforms around 'x' nodes (don't care), this part is usefull for residual networks
    for i in range(2):  # repeat twice to ensure all 'x' nodes are fully optimized
        for node, data in g.nodes(data=True):
            if data['layout'] != 'x':
                continue

            edges = all_edges(g, node)
            edge_tfms = [g.edges[edge]['tfm'] for edge in edges]

            # check if we can reduce the amount of tfms
            if sum(edge_tfms) > len(edges) // 2:
                toggle_tfms(g, edges)

            # if all inputs contain a tfm, move them to the outputs
            edge_in_tfms = [g.edges[edge]['tfm'] for edge in g.in_edges(node)]
            if sum(edge_in_tfms) == len(edge_in_tfms):
                toggle_tfms(g, edges)


def is_op_layout_sensitive(call):
    """Determine if the op in 'call' is sensitive to layout or not.
    For multiple inputs, we assume that all inputs have the same layout.
    Return True in case its sensitive to layout, return False otherwise
    """
    # ops that are not layout sensitive, regardless of their input
    non_sensitive_ops = ['cast', 'clip', 'floor', 'nn.batch_flatten']
    if call.op.name in non_sensitive_ops:
        return False

    # For element-wise ops with two inputs, sensitivity depends on their input type and shape.
    # If at least one input is constant and its not a single value, we consider this op to be
    # sensitive to layout, otherwise not.
    conditional_non_sensitive_ops = ['add', 'multiply', 'divide', 'right_shift', 'concatenate']
    if call.op.name in conditional_non_sensitive_ops:
        for a in call.args:
            if isinstance(a, relay.Constant) and a.data.numpy().size != 1:
                return True
        return False

    # For all other ops that we don't know, consider them sensitive
    return True


class ConstructNetworkXGraph(ExprVisitor):
    """Construct a networkx graph from relay graph
    """
    def __init__(self):
        super().__init__()
        self.g = nx.DiGraph()

    def visit_call(self, call):
        """Convert every op call to a node"""

        # determine layout of this node
        layout = 'x'    # set to don't care
        if isinstance(call.op, relay.Function) and 'Composite' in call.op.attrs:
            if call.op.attrs.Composite != 'soma_dory.add':
                layout = 'acc'
        else:
            # the op is mapped to the cpu. If the op is sensitive to layout, mark that it requires cpu layout.
            if is_op_layout_sensitive(call):
                layout = 'cpu'

        # create node from call
        self.g.add_node(call, layout=layout)

        # add a link from each previous call/node to this one
        for a in call.args:
            if not isinstance(a, relay.Constant):   # avoid creating edges to constants
                self.g.add_edge(a, call, tfm=False)

        self.visit(call.op)
        for a in call.args:
            self.visit(a)

    def visit_function(self, fn):
        # avoid visiting partitioned functions and parameters
        # (partitioned functions have attributes, main function doesn't)
        if fn.attrs is not None:
            return

        for x in fn.params:
            self.visit(x)
        self.visit(fn.body)

        # add output node which is always cpu layout (function return)
        self.g.add_node(fn, layout='cpu')
        self.g.add_edge(fn.body, fn, tfm=False)

    def visit_var(self, var):
        # create node for each input variable (main function only)
        self.g.add_node(var, layout='cpu')


class InsertLayoutTransforms(ExprMutator):
    """Insert layout transforms in relay graph based on information from a networkx graph
    """
    def __init__(self, netx_graph):
        super().__init__()
        self.g = netx_graph

    def visit_function(self, fn):
        # avoid modifying partitioned functions
        if fn.attrs is not None:
            return fn

        new_params = [self.visit(x) for x in fn.params]
        new_body = self.visit(fn.body)

        # add layout transform after last call if needed
        if self.g.edges[(fn.body, fn)]['tfm']:
            new_body = create_layout_transform(new_body, fn.body.checked_type.shape)

        return relay.function.FunctionWithFields(
            fn,
            list(new_params),
            new_body,
        )

    def visit_call(self, call):
        """Insert tfm op before this call if needed for every non constant input
        """
        new_fn = self.visit(call.op)
        new_args = []
        for arg in call.args:
            new_arg = self.visit(arg)
            if not isinstance(arg, relay.Constant) and self.g.edges[(arg, call)]['tfm']:
                new_arg = create_layout_transform(new_arg, arg.checked_type.shape)
            new_args.append(new_arg)

        return relay.Call(new_fn, new_args, call.attrs, call.type_args, call.span)


#import matplotlib.pyplot as plt
#def show_graph(g):
#    pos = nx.spring_layout(g, scale=2)
#    plt.figure(figsize=(12, 12))
#    nx.draw(g, pos, node_size=1000)
#    node_labels = nx.get_node_attributes(g, 'layout')
#    nx.draw_networkx_labels(g, pos, labels=node_labels)
#    edge_labels = nx.get_edge_attributes(g, 'tfm')
#    edge_labels = {k:('tfm' if v else '') for k, v in edge_labels.items()}
#    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
#    plt.show()


@tvm.ir.transform.module_pass(opt_level=0)
class SomaDoryLayoutTransform:
    """ Insert layout transforms where needed
    """
    def transform_module(
        self, mod: tvm.ir.IRModule, ctx: tvm.ir.transform.PassContext
    ) -> tvm.ir.IRModule:
        for global_var, func in mod.functions.items():
            netx_ctor = ConstructNetworkXGraph()
            netx_ctor.visit(func)

            calculate_transforms(netx_ctor.g)
            #show_graph(netx_ctor.g)    # for debugging, uncomment to visualize networkx graph

            tfm = InsertLayoutTransforms(netx_ctor.g)
            func = tfm.visit(func)

            mod.update_func(global_var, func)
        return mod

    def __call__(self, mod):
        return self.transform_module(mod)
