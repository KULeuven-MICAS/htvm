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

"""Soma Dory pattern_table tests"""

import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay.op.contrib import soma_dory


def make_conv2d_pattern(input_shape=[1, 3, 16, 32],
                        weight_shape=[5, 3, 1, 1],
                        conv_channels=5,
                        kernel_size=[1, 1],
                        shift_factor=1,
                        relu=True,
                        bias_add=True,
                        padding=[0, 0, 0, 0],
                        strides=[1, 1],
                        dilation=[1, 1],
                        groups=1,
                        data_layout='NCHW',
                        kernel_layout='OIHW'):

    dtype = 'int8'
    x = relay.var('input', relay.TensorType(input_shape, dtype))
    w = relay.const(np.zeros(weight_shape, dtype))
    x = relay.qnn.op.conv2d(x, w, relay.const(0), relay.const(0),
                            relay.const(1.0), relay.const(1.0),
                            kernel_size,
                            conv_channels,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            data_layout=data_layout,
                            kernel_layout=kernel_layout)
    if bias_add:
        b = relay.const(np.zeros(conv_channels, 'int32'))
        x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_factor))
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, dtype)
    # Optional: ReLU
    if relu:
        x = relay.op.clip(x, a_min=0, a_max=127)

    # make IR module
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    # Infer data types
    mod = tvm.relay.transform.InferType()(mod)

    # extract function body pattern
    pattern = mod['main'].body

    return pattern


def test_check_qnn_conv2d_without_relu():
    pattern = make_conv2d_pattern(relu=False)
    assert soma_dory.check_qnn_conv2d(pattern)


## Tests that verify detection of supported conv2d attribute values

@pytest.mark.parametrize("kernel_size", [[7, 7], [5, 5], [3, 3], [1, 1]])
def test_check_qnn_conv2d_supported_kernel_sizes(kernel_size):

    pattern = make_conv2d_pattern(weight_shape=[5, 3] + kernel_size, kernel_size=kernel_size)
    assert soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("padding", [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
def test_check_qnn_conv2d_supported_padding(padding):

    pattern = make_conv2d_pattern(padding=padding)
    assert soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("strides", [[1, 1], [2, 2]])
def test_check_qnn_conv2d_supported_strides(strides):

    pattern = make_conv2d_pattern(strides=strides)
    assert soma_dory.check_qnn_conv2d(pattern)


## Tests that verify detection of unsupported conv2d attribute values

@pytest.mark.parametrize("kernel_size", [[1, 3], [3, 1], [2, 2]])
def test_check_qnn_conv2d_unsupported_kernel_sizes(kernel_size):

    pattern = make_conv2d_pattern(weight_shape=[5, 3] + kernel_size, kernel_size=kernel_size)
    assert not soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("strides", [[3, 3], [1, 2], [2, 1]])
def test_check_qnn_conv2d_unsupported_strides(strides):

    pattern = make_conv2d_pattern(strides=strides)
    assert not soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("padding", [[2, 2, 2, 2], [1, 0, 1, 0], [0, 1, 0, 1]])
def test_check_qnn_conv2d_unsupported_padding(padding):

    pattern = make_conv2d_pattern(padding=padding)
    assert not soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("dilation", [[2, 2], [1, 2], [2, 1]])
def test_check_qnn_conv2d_unsupported_dilation(dilation):

    pattern = make_conv2d_pattern(dilation=dilation)
    assert not soma_dory.check_qnn_conv2d(pattern)


@pytest.mark.parametrize("shift_factor", [-1, 32])
def test_check_qnn_conv2d_invalid_shift_factor(shift_factor):
    pattern = make_conv2d_pattern(shift_factor=shift_factor)
    assert not soma_dory.check_qnn_conv2d(pattern)


def test_check_qnn_conv2d_no_bias_add():
    # no bias_add is currently not supported
    pattern = make_conv2d_pattern(bias_add=False)
    assert not soma_dory.check_qnn_conv2d(pattern)

# TODO: make this work
#def test_check_qnn_conv2d_unsupported_data_layout():
#    pattern = make_conv2d_pattern(input_shape=[1, 16, 32, 3],
#                                  weight_shape=[5, 3, 1, 1],
#                                  conv_channels=5,
#                                  kernel_size=[1, 1],
#                                  data_layout='NHWC')
#    assert not soma_dory.check_qnn_conv2d(pattern)
