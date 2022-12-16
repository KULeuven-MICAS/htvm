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
    input_shape = (1, 3, 96, 96)
    num_classes = 4     # NOTE: originally 2, but made 4 to support diana accelerator
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, 'int8'))

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, input_shape)

    # 1st layer
    num_filters_1 = 8
    weights_shape = (num_filters_1, input_shape[1], 3, 3)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv1 = relay_soma_conv2d(x, 'conv1', weights, bias, strides=(2, 2), padding=(1, 1), act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_1, input_shape[2]/2, input_shape[3]/2))

    # 2nd layer
    weights_shape = (num_filters_1, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv2 = relay_soma_conv2d(x, 'conv2', weights, bias, padding=(1, 1), groups=num_filters_1, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_1, input_shape[2]/2, input_shape[3]/2))


    
    num_filters_2 = 2*num_filters_1
    weights_shape = (num_filters_2, num_filters_1, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv3 = relay_soma_conv2d(x, 'conv3', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_2, input_shape[2]/2, input_shape[3]/2))


    # 3rd layer
    weights_shape = (num_filters_2, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv4 = relay_soma_conv2d(x, 'conv4', weights, bias, strides=(2, 2), padding=(1, 1), groups=num_filters_2, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_2, input_shape[2]/4, input_shape[3]/4))


    num_filters_3 = 2*num_filters_2
    weights_shape = (num_filters_3, num_filters_2, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv5 = relay_soma_conv2d(x, 'conv5', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_3, input_shape[2]/4, input_shape[3]/4))

    # 4rth layer
    weights_shape = (num_filters_3, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv6 = relay_soma_conv2d(x, 'conv6', weights, bias, padding=(1, 1), groups=num_filters_3, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_3, input_shape[2]/4, input_shape[3]/4))


    weights_shape = (num_filters_3, num_filters_3, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv7 = relay_soma_conv2d(x, 'conv7', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_3, input_shape[2]/4, input_shape[3]/4))

    # 5th layer
    weights_shape = (num_filters_3, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv8 = relay_soma_conv2d(x, 'conv8', weights, bias, strides=(2, 2), padding=(1, 1), groups=num_filters_3, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_3, input_shape[2]/8, input_shape[3]/8))


    num_filters_4 = 2*num_filters_3
    weights_shape = (num_filters_4, num_filters_3, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv9 = relay_soma_conv2d(x, 'conv9', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_4, input_shape[2]/8, input_shape[3]/8))

    # 6th layer
    weights_shape = (num_filters_4, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv10 = relay_soma_conv2d(x, 'conv10', weights, bias, padding=(1, 1), groups=num_filters_4, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_4, input_shape[2]/8, input_shape[3]/8))


    weights_shape = (num_filters_4, num_filters_4, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv11 = relay_soma_conv2d(x, 'conv11', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_4, input_shape[2]/8, input_shape[3]/8))

    # 7th layer
    weights_shape = (num_filters_4, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv12 = relay_soma_conv2d(x, 'conv12', weights, bias, strides=(2, 2), padding=(1, 1), groups=num_filters_4, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_4, input_shape[2]/16, input_shape[3]/16))



    num_filters_5 = 2*num_filters_4
    weights_shape = (num_filters_5, num_filters_4, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv13 = relay_soma_conv2d(x, 'conv13', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    # 8th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv14 = relay_soma_conv2d(x, 'conv14', weights, bias, padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))

   # # ^^ Part 1 - vv Part 2
   # UNCOMMENT ONLY FOR PART 2
   # num_filters_5 = 128
   # input_shape = (1, 3, 96, 96)
   # input_shape_intermediate = (1, 128, 6, 6)
   # num_classes = 4     # NOTE: originally 2, but made 4 to support diana accelerator
   # x = relay.var("input", relay.TensorType(input_shape_intermediate, 'int8'))
   
    weights_shape = (num_filters_5, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv15 = relay_soma_conv2d(x, 'conv15', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    # 9th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv16 = relay_soma_conv2d(x, 'conv16', weights, bias, padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    weights_shape = (num_filters_5, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv17 = relay_soma_conv2d(x, 'conv17', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    # 10th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv18 = relay_soma_conv2d(x, 'conv18', weights, bias, padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    weights_shape = (num_filters_5, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv19 = relay_soma_conv2d(x, 'conv19', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    # 11th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv20 = relay_soma_conv2d(x, 'conv20', weights, bias, padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    weights_shape = (num_filters_5, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv21 = relay_soma_conv2d(x, 'conv21', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    # 12th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv22 = relay_soma_conv2d(x, 'conv22', weights, bias, padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))


    weights_shape = (num_filters_5, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv23 = relay_soma_conv2d(x, 'conv23', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/16, input_shape[3]/16))

# P#art 2 ^^ - vv Part 3
#   num_filters_5 = 128
#   input_shape = (1, 3, 96, 96)
#   input_shape_intermediate = (1, 128, 6, 6)
#   num_classes = 4     # NOTE: originally 2, but made 4 to support diana accelerator
#   x = relay.var("input", relay.TensorType(input_shape_intermediate, 'int8'))
  

   # 13th layer
    weights_shape = (num_filters_5, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv24 = relay_soma_conv2d(x, 'conv24', weights, bias, strides=(2, 2), padding=(1, 1), groups=num_filters_5, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_5, input_shape[2]/32, input_shape[3]/32))


    num_filters_6 = 2*num_filters_5
    weights_shape = (num_filters_6, num_filters_5, 1, 1)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv25 = relay_soma_conv2d(x, 'conv25', weights, bias, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_6, input_shape[2]/32, input_shape[3]/32))


    # 14th layer
    weights_shape = (num_filters_6, 1, 3, 3)
    weights = create_random_array(weights_shape, f'int8')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv26 = relay_soma_conv2d(x, 'conv26', weights, bias, padding=(1, 1), groups=num_filters_6, act=True, shift_bits=4)

    if add_layout_transforms and weight_bits == 2 and not mixed:
        x = relay_soma_layout_transform(x, (1, num_filters_6, input_shape[2]/32, input_shape[3]/32))

  # ###^^ part 3 - part 4 vv
  #  num_filters_6 = 256
  #  input_shape = (1, 3, 96, 96)
  #  input_shape_intermediate = (1, 256, 3, 3)
  #  num_classes = 4     # NOTE: originally 2, but made 4 to support diana accelerator
  #  x = relay.var("input", relay.TensorType(input_shape_intermediate, 'int8'))
    


    weights_shape = (num_filters_6, num_filters_6, 1, 1)
    if mixed:
        weights = create_random_array(weights_shape, f'int8')
    else:
        weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params_conv27 = relay_soma_conv2d(x, 'conv27', weights, bias, act=True, shift_bits=4)
    
    ##^^ part 4 - part 5 vv
    #um_filters_6 = 256
    #nput_shape = (1, 3, 96, 96)
    #nput_shape_intermediate = (1, 256, 3, 3)
    #um_classes = 4     # NOTE: originally 2, but made 4 to support diana accelerator
    # = relay.var("input", relay.TensorType(input_shape_intermediate, 'int8'))
    #
    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_filters_6, 3, 3))

    # avg pool
    x = relay.nn.avg_pool2d(x, (3, 3))
    x = relay.reshape(x, (1, num_filters_6))

    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_filters_6))

    if weight_bits == 8 or mixed:
        weights_shape = (num_classes, num_filters_6)
        weights = create_random_array(weights_shape, 'int8')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense = relay_soma_dense(x, 'dense', weights, bias, act=False, shift_bits=4)
    elif weight_bits == 2:
        x = relay.reshape(x, (1, num_filters_6, 1, 1))
        weights_shape = (num_classes, num_filters_6, 1, 1)
        weights = create_random_array(weights_shape, 'int2')
        bias = create_random_array(weights_shape[0], 'int32')
        x, params_dense = relay_soma_conv2d(x, 'dense', weights, bias, act=False, shift_bits=4)
        x = relay.reshape(x, (1, num_classes))
    if add_layout_transforms:
        x = relay_soma_layout_transform(x, (1, num_classes))

 #     # Dense, for now always 8-bits and on CPU
 #     weights_shape = (num_classes, num_filters_6)
 #     weights = create_random_array(weights_shape, f'int8')
 #     bias = create_random_array(weights_shape[0], f'int32')
 #     w = relay.var("dense.weights", relay.TensorType(weights.shape, weights.dtype))
 #     b = relay.var("dense.bias", relay.TensorType(bias.shape, bias.dtype))
 #     x = relay.nn.dense(x, w, out_dtype="int32")
 #     x = relay.op.nn.bias_add(x, b)
 #     params_dense = {"dense.weights": weights, "dense.bias": bias}

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
    params_conv1.update(params_conv10)
    params_conv1.update(params_conv11)
    params_conv1.update(params_conv12)
    params_conv1.update(params_conv13)
    params_conv1.update(params_conv14)

    params_conv1.update(params_conv15)
#    #
#    #params_conv1 = params_conv15
#
    params_conv1.update(params_conv16)
    params_conv1.update(params_conv17)
    params_conv1.update(params_conv18)
    params_conv1.update(params_conv19)
    params_conv1.update(params_conv20)
    params_conv1.update(params_conv21)
    params_conv1.update(params_conv22)
    params_conv1.update(params_conv23)

#
#   params_conv1 = params_conv24

    params_conv1.update(params_conv24)
    params_conv1.update(params_conv25)
    params_conv1.update(params_conv26)
#
#    params_conv1 = params_conv27
#
    params_conv1.update(params_conv27)

#     #params_conv1 = params_dense
#
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
