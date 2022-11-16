from utils import (
        tvmc_compile_and_unpack,
        relay_soma_layout_transform,
        relay_soma_conv2d,
        relay_soma_dense,
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


def create_model(weight_bits, add_layout_transforms, mixed):
    #input_shape = (1, 1, 49, 10)
    # Use transpose for better performance
    input_shape = (1, 1, 10, 49)
    num_classes = 12
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    num_filters = 64
    #weights_shape = (num_filters, input_shape[1], 10, 4)
    # Use transpose for better performance
    weights_shape = (num_filters, input_shape[1], 4, 10)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    #x, params_conv1 = relay_soma_conv2d(x, 'conv1', weights, bias, strides=(2, 2), padding=(5, 1), act=True, shift_bits=4)
    # Use transpose for better performance
    x, params_conv1 = relay_soma_conv2d(x, 'conv1', weights, bias, strides=(2, 2), padding=(1, 5), act=True, shift_bits=4)

    ## Not necessary anymore if performed on the accelerator
    #if add_layout_transforms:
    #    #x = relay_soma_layout_transform(x, (1, num_filters, 25, 5))
    #    # Use transpose for better performance
    #    x = relay_soma_layout_transform(x, (1, num_filters, 5, 25))
    
    weights_shape = (num_filters, 1, 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv2 = relay_soma_conv2d(x, 'conv2', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv3 = relay_soma_conv2d(x, 'conv3', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv4 = relay_soma_conv2d(x, 'conv4', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv5 = relay_soma_conv2d(x, 'conv5', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv6 = relay_soma_conv2d(x, 'conv6', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv7 = relay_soma_conv2d(x, 'conv7', weights, bias, act=True, shift_bits=4)

    weights_shape = (num_filters, 1, 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv8 = relay_soma_conv2d(x, 'conv8', weights, bias, padding=(1, 1), groups=num_filters, act=True, shift_bits=4)

    weights_shape = (num_filters, num_filters, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv9 = relay_soma_conv2d(x, 'conv9', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms:
        #x = relay_soma_layout_transform(x, (1, num_filters, 25, 5))
        # Use transpose for better performance
        x = relay_soma_layout_transform(x, (1, num_filters, 5, 25))

    # avg pool
    #x = relay.nn.avg_pool2d(x, (25, 5))
    # Use transpose for better performance
    x = relay.nn.avg_pool2d(x, (5, 25))
    x = relay.reshape(x, (1, num_filters))

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_filters))

    weights_shape = (num_classes, num_filters)
    # Hardcode to 8 bits integers, since dense layers are not supported on analog accelerator
    weights = create_random_array(weights_shape, 'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_dense = relay_soma_dense(x, 'dense', weights, bias, act=False, shift_bits=4)

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_classes))

    # Dense, for now always 8-bits and on CPU
    #weights_shape = (num_classes, num_filters)
    #weights = create_random_array(weights_shape, f'int8')
    #bias = create_random_array(weights_shape[0], f'int32')
    #w = relay.var("dense.weights", relay.TensorType(weights.shape, weights.dtype))
    #b = relay.var("dense.bias", relay.TensorType(bias.shape, bias.dtype))
    #x = relay.nn.dense(x, w, out_dtype="int32")
    #x = relay.op.nn.bias_add(x, b)
    #params_dense = {"dense.weights": weights, "dense.bias": bias}

    x = relay.cast(x, 'float32')    # cast needed since softmax in TVM seems not to work with integer inputs
    x = relay.op.nn.softmax(x)

    # combine all params in one dictionary
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
    if args.manual_layout_transform and 'soma_dory' in args.target:
        args.target = args.target.replace('soma_dory', 'soma_dory -layout_transform=0')
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
