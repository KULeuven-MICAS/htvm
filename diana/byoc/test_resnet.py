import relay_resnet20
from compare_outputs import get_gdb_output
from tvm.driver.tvmc.model import TVMCModel
from utils import (
                   tvmc_compile_and_unpack,
                   create_demo_file,
                   adapt_gcc_opt,
                   make,
                   gdb
                  )
import numpy as np

import tvm.relay as relay
from relay_resnet20 import relay_res_layer, relay_res_block
from utils import relay_soma_conv2d
import tvm


def create_model():
    # initial setup
    input_shape = (1,3,32,32)
    #input_shape = (1,16,16,16)
    params = {}
    input_tensor = relay.var("input", relay.TensorType(input_shape, 'int8'))

    # Connect resnet blocks
    x, params_out = relay_soma_conv2d(input_tensor,
                "conv1",
                (16,3,3,3), np.ones((16,3,3,3), dtype=np.int8),
                np.ones(16, dtype=np.int32))
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_16", 16, 16, True)
    #int_output, params_out = relay_res_layer(x, "res_layer_16", 16, 16, True)
    params.update(params_out)

    #strided = False
    #name = "FAIL"
    ## 3x3 convolution block 1
    #weights_shape = (32,16,3,3)
    ##x, params_out = relay_soma_conv2d(input_tensor, 
    #x, params_out = relay_soma_conv2d(int_output, 
    #                    name + '_3x3conv_0',
    #                    weights_shape, np.ones(weights_shape, dtype=np.int8),
    #                    np.ones(32, dtype=np.int32),
    #                    strides=(1,1))
    #params.update(params_out)
    ## 3x3 convolution block 2
    #weights_shape = (32,32,3,3)
    #x, params_out = relay_soma_conv2d(x, 
    #                    name + '_3x3conv_1',
    #                    weights_shape, np.ones(weights_shape, dtype=np.int8),
    #                    np.ones(32, dtype=np.int32),
    #                    strides=((2,2) if strided else (1,1)))
    #params.update(params_out)
    # 1x1 skip layer convolution
    #weights_shape = (32,16,1,1)
    #y, params_out = relay_soma_conv2d(int_output, 
    #                    name + '_1x1conv',
    #                    weights_shape, np.ones(weights_shape, dtype=np.int8),
    #                    np.ones(32, dtype=np.int32),
    #                    strides=((2,2) if strided else (1,1)))
    #params.update(params_out)
    #x = y
    #x = relay.add(x, y)


    #x, params_out = relay_res_block(input_tensor, "FAIL", 16, 32, False)
    #x, params_out = relay_res_block(x, "FAIL", 16, 32, False)
    #params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_32", 16, 32, True)
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_64", 32, 64, False)
    params.update(params_out)

    x = relay.nn.avg_pool2d(x, (8,8))
    x = relay.reshape(x, (1,64))

    fc_weights_name = "fc_weights"
    fc_weights_shape = (10,64)
    fc_weights = relay.var(fc_weights_name, 
                           relay.TensorType(fc_weights_shape, "int8"))
    params.update({fc_weights_name: tvm.nd.array(np.ones(fc_weights_shape, 
                                                          dtype=np.int8))})
    x = relay.nn.dense(x, fc_weights, out_dtype="int8")
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params


#import resnet20 model
mod, params = create_model()
model = TVMCModel(mod, params)
#init_value = -2
init_value = 1

# run on X86 to get demo_x86.txt
print("TEST: Running on X86")
device = "x86"
target = "c"
fusion = False
tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
create_demo_file(mod, init_value=init_value)
adapt_gcc_opt("Makefile.x86", 0)
make(device)
gdb(device, "build/demo", "gdb_demo_x86.sh")
print("TEST: parsing X86 output")

demo_x86 = get_gdb_output("demo_x86.txt")

# run on X86 to get demo_x86.txt
print("TEST: Running on Diana")
device = "pulp"
target = "soma_dory, c"
fusion = True
tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
create_demo_file(mod, init_value=init_value)
adapt_gcc_opt("Makefile.pulprt", 3)
make(device)
gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
print("TEST: parsing PULP output")

demo_pulp = get_gdb_output("demo.txt")

if np.ma.allequal(demo_x86,demo_pulp):
    print("TEST: PASS")
else:
    print("TEST: FAIL")
