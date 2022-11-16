from utils import (
        tvmc_compile_and_unpack,
        relay_soma_layout_transform,
        relay_soma_conv2d,
        relay_soma_dense,
        relay_soma_add,
        create_demo_file,
        parse_cli_options,
        create_random_array
        )
from profiler import insert_profiler
import os
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import numpy as np


def create_model(weight_bits, add_layout_transforms, mixed=False):
    input_shape = (1, 3, 32, 32)
    num_classes = 12    # NOTE originally 10, but made 12 to support diana accelerator
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    # Assume incoming data layout is already correct
    #if add_layout_transforms:
    #   x = relay_soma_layout_transform(x, input_shape)

    num_filters_1 = 16
    weights_shape = (num_filters_1, 3, 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv1 = relay_soma_conv2d(x, 'conv1', weights, bias, padding=(1, 1), act=True, shift_bits=4)

    # First stack
    weights_shape = (num_filters_1, num_filters_1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv2 = relay_soma_conv2d(x, 'conv2', weights, bias, padding=(1, 1), act=True, shift_bits=4)

    weights_shape = (num_filters_1, num_filters_1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv3 = relay_soma_conv2d(y, 'conv3', weights, bias, padding=(1, 1), act=False, shift_bits=4)

    x = relay_soma_add(x, y, "add_1", act=True)

    # Second stack
    num_filters_2 = 32
    weights_shape = (num_filters_2, num_filters_1, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv4 = relay_soma_conv2d(x, 'conv4', weights, bias, strides=(2, 2), padding=(1, 1), act=True, shift_bits=4)

    weights_shape = (num_filters_2, num_filters_2, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv5 = relay_soma_conv2d(y, 'conv5', weights, bias, padding=(1, 1), act=False, shift_bits=4)

    weights_shape = (num_filters_2, num_filters_1, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv6 = relay_soma_conv2d(x, 'conv6', weights, bias, strides=(2, 2), act=False, shift_bits=4)

    x = relay_soma_add(x, y, "add_2", act=True)

    # Third stack
    num_filters_3 = 64
    weights_shape = (num_filters_3, num_filters_2, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv7 = relay_soma_conv2d(x, 'conv7', weights, bias, strides=(2, 2), padding=(1, 1), act=True, shift_bits=4)

    weights_shape = (num_filters_3, num_filters_3, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    y, params_conv8 = relay_soma_conv2d(y, 'conv8', weights, bias, padding=(1, 1), act=False, shift_bits=4)

    weights_shape = (num_filters_3, num_filters_2, 1, 1)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv9 = relay_soma_conv2d(x, 'conv9', weights, bias, strides=(2, 2), act=False, shift_bits=4)

    x = relay_soma_add(x, y, "add_3", act=True)

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_filters_3, 8, 8))

    # avg pool
    x = relay.nn.avg_pool2d(x, (8,8))
    x = relay.reshape(x, (1, num_filters_3))

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_filters_3))

    if weight_bits == 8 or mixed:
        weights_shape = (num_classes, num_filters_3)
        weights = create_random_array(weights_shape, 'int8')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense = relay_soma_dense(x, 'dense', weights, bias, act=False, shift_bits=4)
    elif weight_bits == 2:
        x = relay.reshape(x, (1, num_filters_3, 1, 1))
        weights_shape = (num_classes, num_filters_3, 1, 1)
        weights = create_random_array(weights_shape, 'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense = relay_soma_conv2d(x, 'dense', weights, bias, padding=(0,0), act=False, shift_bits=4)


    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_classes))

    # Dense, for now always 8-bits and on CPU
    #weights_shape = (num_classes, num_filters_3)
    #weights = create_random_array(weights_shape, f'int8')
    #bias = create_random_array(weights_shape[0], f'int32')
    #w = relay.var("dense.weights", relay.TensorType(weights.shape, weights.dtype))
    #b = relay.var("dense.bias", relay.TensorType(bias.shape, bias.dtype))
    #x = relay.nn.dense(x, w, out_dtype="int32")
    #x = relay.op.nn.bias_add(x, b)
    #params_dense = {"dense.weights": weights, "dense.bias": bias}

    x = relay.cast(x, 'float32')    # cast needed since softmax in TVM seems not to work with integer inputs
    x = relay.op.nn.softmax(x)

    ## combine all params in one dictionary
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
    # for reproducability
    np.random.seed(0)
    # Get options from cli
    args, opt_string = parse_cli_options()

    add_layout_transforms = False
    if '-layout_transform=0' in args.target:
        add_layout_transforms = True

    # create the model
    mod, params = create_model(args.weight_bits, add_layout_transforms)
    model = TVMCModel(mod, params)

    # compile the model
    tvmc_compile_and_unpack(model, target=args.target, fuse_layers=args.fusion)
    create_demo_file(mod, path='../src/demo.c')
    basename_this_file = os.path.splitext(os.path.basename(__file__))[0]
    insert_profiler(measurement=args.measurement,
                    interactive=args.interactive,
                    csv_file=f"{basename_this_file}_{opt_string}.csv")
