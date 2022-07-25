from utils import tvmc_compile_and_unpack
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import numpy as np


def create_int8_conv_bias_act(x, name, weights_shape, act=False, shift_bits=0):
    # define weights and bias variables
    weights_name = name + '.weights'
    bias_name = name + '.bias'
    conv_channels = weights_shape[0]
    w = relay.var(weights_name, relay.TensorType(weights_shape, 'int8'))
    b = relay.var(bias_name, relay.TensorType((conv_channels,), 'int32'))

    # define weights and bias values
    #w_value = np.random.uniform(low=-10, high=10, size=weights_shape).astype(np.int8)
    w_value = np.ones(weights_shape).astype(np.int8)
    #b_value = np.random.uniform(low=-10, high=10, size=conv_channels).astype(np.int32)
    b_value = np.ones(conv_channels).astype(np.int32)
    params = {weights_name: tvm.nd.array(w_value), bias_name: tvm.nd.array(b_value)}

    # define ops
    x = relay.qnn.op.conv2d(x, w, relay.const(0), relay.const(0), relay.const(1.0), relay.const(1.0), weights_shape[-2:], channels=conv_channels, padding=(1, 1))
    x = relay.op.nn.bias_add(x, b)
    #x = relay.op.right_shift(x, relay.const(shift_bits)) 
    #x = relay.op.cast(x, 'int8')
    x = relay.qnn.op.requantize(x, relay.const(1.0), relay.const(0), relay.const(float(2**shift_bits)), relay.const(0), axis=1, out_dtype='int8')
    # the fourth param of requantize contains the power-of-two division factor. All other constants will be ignored by soma codegen


    if act:
        x = relay.op.clip(x, a_min=-128, a_max=127)

    return x, params


def create_model():
    input_shape = (1, 3, 32, 32)
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    weights_shape = (16, 3, 3, 3)
    x, params1 = create_int8_conv_bias_act(x, 'conv1', weights_shape, True, 2)

    #weights_shape = (32, 16, 3, 3)
    #x, params2 = create_int8_conv_bias_act(x, 'conv2', weights_shape, True, 4)

    #weights_shape = (16, 32, 3, 3)
    #x, params3 = create_int8_conv_bias_act(x, 'conv3', weights_shape, True, 4)

    # combine all params
    #params1.update(params2)
    #params1.update(params3)
    params = params1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    return mod, params

# create the model
mod, params = create_model()

# compile the model
model = TVMCModel(mod, params)
tvmc_compile_and_unpack(model, target="c", fuse_layers=True)
