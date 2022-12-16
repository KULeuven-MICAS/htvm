from utils import (
        tvmc_compile_and_unpack,
        relay_soma_layout_transform,
        relay_soma_dense,
        relay_soma_conv2d,
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
    input_shape = (1, 640)
    num_outputs = input_shape[1]
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, 'int8'))

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, input_shape)

    num_units = 128

    if mixed:
        weights_shape = (num_units, input_shape[1])
        weights = create_random_array(weights_shape, f'int8')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense1 = relay_soma_dense(x, 'dense1', weights, bias, act=True, shift_bits=4)

        x = relay.reshape(x, (1, 128, 1, 1))
        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense2 = relay_soma_conv2d(x, 'dense2', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense3 = relay_soma_conv2d(x, 'dense3', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense4 = relay_soma_conv2d(x, 'dense4', weights, bias, padding=(0,0), act=True, shift_bits=4)

    elif weight_bits == 8:
        weights_shape = (num_units, input_shape[1])
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense1 = relay_soma_dense(x, 'dense1', weights, bias, act=True, shift_bits=4)


        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense2 = relay_soma_dense(x, 'dense2', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense3 = relay_soma_dense(x, 'dense3', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense4 = relay_soma_dense(x, 'dense4', weights, bias, act=True, shift_bits=4)
 
    elif weight_bits == 2:
        # Workaround - implement dense as conv2d on analog acc
        x = relay.reshape(x, (input_shape[0],input_shape[1], 1, 1))
        weights_shape = (num_units, input_shape[1], 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense1 = relay_soma_conv2d(x, 'dense1', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense2 = relay_soma_conv2d(x, 'dense2', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense3 = relay_soma_conv2d(x, 'dense3', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense4 = relay_soma_conv2d(x, 'dense4', weights, bias, padding=(0,0), act=True, shift_bits=4)
    else:
        raise NotImplementedError

    num_units_neck = 8

    if mixed:
        weights_shape = (num_units_neck, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense5 = relay_soma_conv2d(x, 'dense5', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units_neck, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense6 = relay_soma_conv2d(x, 'dense6', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense7 = relay_soma_conv2d(x, 'dense7', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense8 = relay_soma_conv2d(x, 'dense8', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense9 = relay_soma_conv2d(x, 'dense9', weights, bias, padding=(0,0), act=True, shift_bits=4)

        x = relay.reshape(x, (1,num_units))
        weights_shape = (num_outputs, num_units)
        weights = create_random_array(weights_shape, f'int8')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense10 = relay_soma_dense(x, 'dense10', weights, bias, act=False, shift_bits=4)


    elif weight_bits == 8:
        weights_shape = (num_units_neck, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense5 = relay_soma_dense(x, 'dense5', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units_neck)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense6 = relay_soma_dense(x, 'dense6', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense7 = relay_soma_dense(x, 'dense7', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense8 = relay_soma_dense(x, 'dense8', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_units, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense9 = relay_soma_dense(x, 'dense9', weights, bias, act=True, shift_bits=4)

        weights_shape = (num_outputs, num_units)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense10 = relay_soma_dense(x, 'dense10', weights, bias, act=False, shift_bits=4)

    elif weight_bits == 2:
        weights_shape = (num_units_neck, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense5 = relay_soma_conv2d(x, 'dense5', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units_neck, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense6 = relay_soma_conv2d(x, 'dense6', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense7 = relay_soma_conv2d(x, 'dense7', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense8 = relay_soma_conv2d(x, 'dense8', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_units, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense9 = relay_soma_conv2d(x, 'dense9', weights, bias, padding=(0,0), act=True, shift_bits=4)

        weights_shape = (num_outputs, num_units, 1, 1)
        weights = create_random_array(weights_shape, f'int{weight_bits}')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense10 = relay_soma_conv2d(x, 'dense10', weights, bias, padding=(0,0), act=False, shift_bits=4)
    else:
        raise NotImplementedError


    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_outputs))

    params_dense1.update(params_dense2)
    params_dense1.update(params_dense3)
    params_dense1.update(params_dense4)
    params_dense1.update(params_dense5)
    params_dense1.update(params_dense6)
    params_dense1.update(params_dense7)
    params_dense1.update(params_dense8)
    params_dense1.update(params_dense9)
    params_dense1.update(params_dense10)
    params = params_dense1

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
