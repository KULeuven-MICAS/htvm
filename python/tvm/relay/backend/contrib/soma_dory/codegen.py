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
"""Codegen for DORY SOMA"""

import importlib
import numpy as np

# TVM imports
import tvm
from tvm.relay.dataflow_pattern import is_op
from tvm.relay.expr_functor import ExprVisitor
from . import _ffi_api

# DORY imports
from dory.Parsers.Layer_node import Layer_node
from dory.Parsers.DORY_node import DORY_node
from dory.Hardware_targets.Diana.Diana_TVM.HW_Parser import onnx_manager
from dory.Hardware_targets.Diana.Diana_TVM.C_Parser import C_Parser


def get_root_call(call, root_op_name):
    if not isinstance(call, tvm.relay.Call):
        return None
    if str(call.op) == root_op_name:
        return call
    return get_root_call(call.args[0], root_op_name)


def borders(bits, signed):
    low = -(2 ** (bits-1)) if signed else 0
    high = 2 ** (bits-1) - 1 if signed else 2 ** bits - 1
    return low, high


def tvm_array_to_list(array):
    """TVM wraps a lists in tvm.ir.container.Array containers and integers in tvm.tir.expr.IntImm objects
    This function converts a tvm array with IntImm values to a regular python list with integers
    """
    return [int(v) for v in array]


def create_dory_conv_node(call, index: int, relu: bool, is_analog: bool):
    """Populate a dory layer node with convolution args and attrs
    """
    conv_call = get_root_call(call.op.body, "nn.conv2d")
    right_shift_call = get_root_call(call.op.body, "right_shift")

    # TODO: assert that weights and bias are constants
    input_dims = call.args[0].type_annotation.shape
    output_dims = right_shift_call.args[0].checked_type.shape
    weights = call.args[1].data

    if is_analog:
        mul_call = get_root_call(call.op.body, "multiply")
        add_call = get_root_call(call.op.body, "add")
        div_call = get_root_call(call.op.body, "divide")
        bn_weight = mul_call.args[1].data
        bn_bias = add_call.args[1].data
    else:
        bias = call.args[2].data

    shift_value = right_shift_call.args[1].data

    node = Layer_node()
    node.name = 'Convolution'
    node.op_type = 'Conv'
    node.pads = tvm_array_to_list(conv_call.attrs.padding)
    node.group = int(conv_call.attrs.groups)
    node.strides = tvm_array_to_list(conv_call.attrs.strides)
    node.kernel_shape = tvm_array_to_list(conv_call.attrs.kernel_size)
    node.input_dimensions = tvm_array_to_list(input_dims[-2:])
    node.output_dimensions = tvm_array_to_list(output_dims[-2:])
    node.input_channels = int(input_dims[1])
    node.output_channels = int(output_dims[1])

    if relu:
        node.output_activation_type = 'uint'
    else:
        node.output_activation_type = 'int'

    node.output_activation_bits = 7 if is_analog else 8
    node.input_activation_type = 'int'
    node.input_activation_type = 'int'
    node.input_activation_bits = 8
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = 8
    node.weight_type = 'int'
    node.weight_bits = 2 if is_analog else 8
    node.bias_bits = 32
    node.add_memory_and_MACs()

    node.number_of_input_nodes = 1
    node.input_indexes = [str(index)]
    node.output_index = str(index + 1)
    node.number_of_input_constants = 1
    node.constant_names.append('weights')
    node.weights = {
        'value': weights.numpy(),
        'layout': 'CoutCinK'
    }

    if is_analog:
        node.constant_names.append('k')
        node.k = {
            'value': bn_weight.numpy().flatten(),
            'layout': ''
        }
        node.constant_names.append('l')
        node.l = {
            'value': bn_bias.numpy().flatten(),
            'layout': ''
        }
    else:
        node.constant_names.append('bias')
        node.bias = {
            'value': bias.numpy(),
            'layout': ''
        }

    node.constant_names.append('outshift')
    node.outshift = {
        'value': shift_value,
        'layout': ''
    }

    return node


# TODO: merge create_dory_dense_node and create_dory_conv2d_node
def create_dory_dense_node(call, index: int, relu: bool = False):
    """Populate a dory layer node with fully-connected/dense args and attrs
    """
    if len(call.args) != 3:
        raise ValueError(f"Expected number of args for doma_dory.dense is 3, got {len(call.args)}")

    dense_call = get_root_call(call.op.body, "nn.dense")
    right_shift_call = get_root_call(call.op.body, "right_shift")

    # TODO: assert that weights and bias are constants
    input_dims = call.args[0].type_annotation.shape
    output_dims = right_shift_call.args[0].checked_type.shape
    weights = call.args[1].data
    bias = call.args[2].data
    shift_value = right_shift_call.args[1].data

    assert weights.dtype[:3] == 'int', "Expected weights to be of type intX"

    node = Layer_node()
    node.name = 'FullyConnected'
    node.op_type = 'Gemm'
    node.pads = [0, 0, 0, 0]
    node.group = 1
    node.strides = [1, 1]
    node.kernel_shape = [1, 1]
    node.input_dimensions = [1, 1]
    node.output_dimensions = [1, 1]
    node.input_channels = int(input_dims[1])
    node.output_channels = int(output_dims[1])

    if relu:
        node.output_activation_type = 'uint'
    else:
        node.output_activation_type = 'int'
    node.output_activation_bits = 8
    node.input_activation_type = 'int'
    node.input_activation_bits = 8
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None
    node.weight_type = 'int'
    node.weight_bits = int(weights.dtype[3:])   # extract the bit number from the dtype
    node.bias_bits = 128
    node.add_memory_and_MACs()

    node.number_of_input_nodes = 1
    node.input_indexes = [str(index)]
    node.output_index = str(index + 1)
    node.number_of_input_constants = 1
    node.constant_names.append('weights')
    w = weights.numpy()
    w = w.reshape(*w.shape, 1, 1)
    node.weights = {
        'value': w,
        'layout': 'CoutCinK'
    }
    node.constant_names.append('bias')
    node.bias = {
        'value': bias.numpy(),
        'layout': ''
    }
    node.constant_names.append('outshift')
    node.outshift = {
        'value': shift_value,
        'layout': ''
    }

    return node


def create_dory_add_node(call, index_1: int, index_2: int, index_out: int, relu: bool = False):
    add_call = get_root_call(call.op.body, "add")
    right_shift_call = get_root_call(call.op.body, "right_shift")

    # TODO: assert that weights and bias are constants
    input_dims = call.args[0].type_annotation.shape
    output_dims = right_shift_call.args[0].checked_type.shape
    shift_value = right_shift_call.args[1].data


    assert call.args[0].type_annotation.dtype[:3] == 'int', "Expected weights to be of type intX"

    node = Layer_node()
    node.name = 'Addition'
    node.op_type = 'Add' # TODO might be redundant
    node.pads = [0, 0, 0, 0]
    node.group = 1
    node.strides = [1, 1]
    node.kernel_shape = [1,1]
    node.input_dimensions = tvm_array_to_list(input_dims[-2:])
    node.output_dimensions = tvm_array_to_list(input_dims[-2:])
    node.input_channels = int(input_dims[-3])
    node.output_channels = int(input_dims[-3])
    if relu:
        node.output_activation_type = 'uint'
    else:
        node.output_activation_type = 'int'
    node.output_activation_bits = 8
    node.input_activation_type = 'int'
    node.input_activation_bits = 8
    node.constant_names = []
    node.constant_names.append('inmul1')
    node.inmul1 = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('inadd1')
    node.inadd1 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inshift1')
    node.inshift1 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inmul2')
    node.inmul2 = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('inadd2')
    node.inadd2 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('inshift2')
    node.inshift2 = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('outmul')
    node.outmul = {
        'value': 1,
        'layout': ''
    }
    node.constant_names.append('outadd')
    node.outadd = {
        'value': 0,
        'layout': ''
    }
    node.constant_names.append('outshift')
    node.outshift = {
        'value': shift_value,
        'layout': ''
    }
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None
    node.weight_type = 'int'
    node.weight_bits = int(call.args[0].checked_type.dtype[3:])
    node.bias_bits = 32
    node.weight_memory = None
    node.MACs = node.output_dimensions[0] * node.output_dimensions[1] * node.output_channels \
                * node.kernel_shape[1] * node.kernel_shape[0] * node.input_channels
    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = [str(index_1), str(index_2)]  # '0' is the network input
    node.output_index = str(index_out)
    # Constants: weights
    node.number_of_input_constants = 1
    # TODO: Make this parameter based on Relay graph to avoid unnecessary data transfer
    # Set branch out to 1 to enable DMA transfer
    node.branch_out = 1
    return node


class RelayToDoryGraph(ExprVisitor):
    """Convert relay graph to dory graph
    """
    def __init__(self):
        super().__init__()
        self.dory_graph = []

    def visit_call(self, call):
        """Extract parameters and construct dory graph
        """
        if not isinstance(call.op, tvm.relay.Function):
            raise ValueError(f"Expected call.op to be relay.Function, got {type(call.op)}")

        pattern_name = call.op.attrs['Composite']
        clip_call = call.op.body.args[0]

        # if clip_call.attrs.a_min == 0.0 --> invoke relu operation
        use_relu = clip_call.attrs.a_min == 0.0

        if pattern_name == 'soma_dory.aconv2d':
            self.dory_graph.append(create_dory_conv_node(call, 0, use_relu, True))

        elif pattern_name == 'soma_dory.conv2d':
            self.dory_graph.append(create_dory_conv_node(call, 0, use_relu, False))

        elif pattern_name == 'soma_dory.dense':
            self.dory_graph.append(create_dory_dense_node(call, 0, use_relu))

        elif pattern_name == 'soma_dory.add':
            self.dory_graph.append(create_dory_add_node(call, 0, 1, 2, use_relu))

        else:
            raise ValueError(f"Unknown composite function {pattern_name}")


def formatting_constant_parameters_tensors(hw_graph):
    """Copied from DORY HW_Parser.py, but without calculating the checksums for the activations
    """
    for i, node in enumerate(hw_graph):
        if node.get_parameter('weight_bits') >= 8:
            node.add_checksum_w_integer()
        else:
            weight_name = ""
            if "Convolution" in node.name or "FullyConnected" in node.name:
                for name in node.constant_names:
                    if name not in ["l","k","outshift","outmul","outadd"]:
                        if "bias" not in name:
                            weight_name = name
            if weight_name in node.__dict__:
                node.__dict__[weight_name]["value"] = node.__dict__[weight_name]["value"].flatten().tolist()
            node.check_sum_w = 0


@tvm._ffi.register_func("relay.ext.soma_dory")
def soma_dory_compiler(mod: tvm.ir.IRModule):
    """Our codegen entry point
    mod:        IRModule of partitioned relay graph
    returns:    C code wrapped in a runtime module
    """

    codegen = RelayToDoryGraph()
    codegen.visit(mod)

    # Low level DORY passes
    config_file = {'onnx_file': '', 'code reserved space': 0, 'BNRelu_bits': 32}
    converter = onnx_manager(codegen.dory_graph, config_file, '')
    converter.mapping_to_HW_nodes()
    converter.adjust_data_layout()
    converter.add_tensors_memory_occupation_and_MACs()
    converter.tiling()
    converter.renaming_weights()
    hw_graph = converter.DORY_Graph

    formatting_constant_parameters_tensors(hw_graph)

    # use function name that tvm provides
    func_name = mod.attrs.global_symbol
    hw_graph[0].name = func_name
    syms = [func_name]

    # append MACs to MACs report, this is dirty
    with open('/tmp/macs_report.txt', 'a') as f:
        f.write(f"{func_name},{hw_graph[0].MACs},{hw_graph[0].weight_memory}\n")

    generator = C_Parser(hw_graph, config_file, '', 'None', 'No', 'auto', '')
    code_strings = generator.mapping_layers_to_C_files()
    code = code_strings[0]  # we only expect a single string containing the generated code

    return _ffi_api.CSourceModuleCreate(code, "c", syms, [])
