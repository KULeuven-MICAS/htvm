from utils import (
        tvmc_compile_and_unpack,
        relay_soma_conv2d,
        create_demo_file,
        parse_cli_options,
        create_random_array
        )
from profiler import insert_profiler
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import numpy as np

# for reproducability
np.random.seed(0)

def create_model(weight_bits):
    input_shape = (1, 1, 49, 10)
    num_classes = 12
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    num_filters = 64
    weights_shape = (num_filters, input_shape[1], 10, 4)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv1 = relay_soma_conv2d(x, 'conv1', weights, bias, strides=(2, 2), padding=(5, 1), act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv2 = relay_soma_conv2d(x, 'conv2', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv3 = relay_soma_conv2d(x, 'conv3', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv4 = relay_soma_conv2d(x, 'conv4', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv5 = relay_soma_conv2d(x, 'conv5', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv6 = relay_soma_conv2d(x, 'conv6', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv7 = relay_soma_conv2d(x, 'conv7', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv8 = relay_soma_conv2d(x, 'conv8', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv9 = relay_soma_conv2d(x, 'conv9', weights, bias, act=True, shift_bits=4)

    # avg pool
    x = relay.nn.avg_pool2d(x, (25, 5))
    x = relay.reshape(x, (1, num_filters))

    # Dense, for now always 8-bits and on CPU
    weights_shape = (num_classes, num_filters)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], f'int32')
    w = relay.var("dense.weights", relay.TensorType(weights.shape, weights.dtype))
    b = relay.var("dense.bias", relay.TensorType(bias.shape, bias.dtype))
    x = relay.nn.dense(x, w, out_dtype="int32")
    x = relay.op.nn.bias_add(x, b)
    params_dense = {"dense.weights": weights, "dense.bias": bias}

    x = relay.cast(x, 'float32')    # cast needed since softmax in TVM seems not to work with integer inputs
    x = relay.op.nn.softmax(x)

    ### combine all params in one dictionary
    params_conv1.update(params_conv2)
    params_conv1.update(params_conv3)
    params_conv1.update(params_conv4)
    params_conv1.update(params_conv5)
    params_conv1.update(params_conv6)
    params_conv1.update(params_conv7)
    params_conv1.update(params_conv8)
    params_conv1.update(params_conv9)
    params_conv1.update(params_dense)
    params = params_conv1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    return mod, params


if __name__ == "__main__":
    target, measurement, interactive, fusion, weight_bits, gcc_opt = parse_cli_options()
    # create the model
    mod, params = create_model(weight_bits)
    model = TVMCModel(mod, params)
    # compile the model
    tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
    create_demo_file(mod, path='../src/demo.c')
    fusion_name = "fused" if fusion else "unfused"
    target_name = "dory" if target == "soma_dory, c" else "c"
    csv_name = f"relay_simple_{target_name}_{fusion_name}" + \
               f"_O{gcc_opt}_{measurement}.csv"
    insert_profiler(measurement=measurement,
                    interactive=interactive,
                    csv_file=csv_name)
