from utils import (
                   tvmc_compile_and_unpack, 
                   relay_soma_conv2d, 
                   create_demo_file,
                   parse_cli_options
                  )
from profiler import insert_profiler
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend import Executor, Runtime
import numpy as np

def relay_soma_add(a, b):
    a = relay.op.cast(a, 'int16')
    b = relay.op.cast(b, 'int16')
    x = relay.op.add(a, b)
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')
    return x



def relay_res_block(input_tensor, name, input_channels, output_channels,
                    strided=True):
    # initial setup
    params = {}
    # 3x3 convolution block 1
    weights_shape = (output_channels,input_channels,3,3)
    x, params_out = relay_soma_conv2d(input_tensor, 
                        name + '_3x3conv_0',
                        weights_shape, np.ones(weights_shape, dtype=np.int8),
                        np.ones(output_channels, dtype=np.int32),
                        strides=(1,1))
    params.update(params_out)
    # 3x3 convolution block 2
    weights_shape = (output_channels,output_channels,3,3)
    x, params_out = relay_soma_conv2d(x, 
                        name + '_3x3conv_1',
                        weights_shape, np.ones(weights_shape, dtype=np.int8),
                        np.ones(output_channels, dtype=np.int32),
                        strides=((2,2) if strided else (1,1)))
    params.update(params_out)
    # 1x1 skip layer convolution
    weights_shape = (output_channels,input_channels,1,1)
    y, params_out = relay_soma_conv2d(input_tensor, 
                        name + '_1x1conv',
                        weights_shape, np.ones(weights_shape, dtype=np.int8),
                        np.ones(output_channels, dtype=np.int32),
                        strides=((2,2) if strided else (1,1)))
    params.update(params_out)
    #x = relay_soma_add(x, y)
    x = relay.add(x, y)
    return x, params

def relay_res_layer(input_tensor, name, input_channels, output_channels, strided):
    params = {}
    x, params_out = relay_res_block(input_tensor, name + "block_1", input_channels,
                                    output_channels, False) 
    params.update(params_out)
    x, params_out = relay_res_block(x, name + "block_2", output_channels,
                                    output_channels, False) 
    params.update(params_out)
    x, params_out = relay_res_block(x, name + "block_3", output_channels,
                                    output_channels, strided) 
    params.update(params_out)
    return x, params

    
def relay_res_layer_broken(input_tensor, name, input_channels, output_channels,
                    repeat_in_block, repeat_block, stride_in_last=True):
    # initial setup
    params = {}
    input_tmp = input_tensor
    for i in range(repeat_block):
        # First write 3x3 convolution blocks
        for j in range(repeat_in_block):
            # First input shape has different weights_shape
            if i == 0 and j==0:
                weights_shape = (output_channels,input_channels,3,3)
                x, params_out = relay_soma_conv2d(input_tensor, 
                        name + '_3x3conv_'+
                        str(i) + "_" + str(j),
                        weights_shape, np.ones(weights_shape, dtype=np.int8),
                        np.zeros(output_channels, dtype=np.int32))
            else:
                strides = (1,1)
                # in the very last 3x3 of one block the stride should be (2,2)
                if(stride_in_last and (i == repeat_block - 1 and j == repeat_in_block - 1)):
                    strides = (2,2)
                weights_shape = (output_channels, output_channels,3,3)
                x, params_out = relay_soma_conv2d(x, 
                        name + '_3x3conv_'+
                        str(i) + "_" + str(j),
                        weights_shape, np.ones(weights_shape, dtype=np.int8),
                        np.zeros(output_channels, dtype=np.int32),
                        strides=strides)
            params.update(params_out)

        # Perform residual convolution (1x1)
        if i == 0:
            weights_shape = (output_channels,input_channels,1,1)
            #print(weights_shape)
        else:
            weights_shape = (output_channels,output_channels,1,1)
            #print(weights_shape)
        if ((i != repeat_block - 1) or (not (stride_in_last))):
            y, params_out = relay_soma_conv2d(input_tmp,
                    name + "_1x1conv_" + str(i),
                    weights_shape, np.ones(weights_shape, dtype=np.int8),
                    np.zeros(output_channels, dtype=np.int32),
                    strides=(1,1))
        else: 
            y, params_out = relay_soma_conv2d(input_tmp,
                    name + "_1x1conv_" + str(i),
                    weights_shape, np.ones(weights_shape, dtype=np.int8),
                    np.zeros(output_channels, dtype=np.int32),
                    strides=(2,2))

        params.update(params_out)
        # Add residual
        x = relay.add(x, y)
        # Save the addition for later use by 1x1 convolution
        input_tmp = x
        
    return x, params

def create_model():
    # initial setup
    input_shape = (1,3,32,32)
    params = {}
    input_tensor = relay.var("input", relay.TensorType(input_shape, 'int8'))

    # Connect resnet blocks
    x, params_out = relay_soma_conv2d(input_tensor,
                "conv1",
                (16,3,3,3), np.ones((16,3,3,3), dtype=np.int8),
                np.ones(16, dtype=np.int32))
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_16", 16, 16, True)
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_32", 16, 32, True)
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_64", 32, 64, False)
    params.update(params_out)

    x = relay.nn.avg_pool2d(x, (8,8))
    x = relay.reshape(x, (1,64))

    #fc_weights_name = "fc_weights"
    #fc_weights_shape = (10,64)
    #fc_weights = relay.var(fc_weights_name, 
    #                       relay.TensorType(fc_weights_shape, "int8"))
    #params.update({fc_weights_name: tvm.nd.array(np.ones(fc_weights_shape, 
    #                                                      dtype=np.int8))})
    #x = relay.nn.dense(x, fc_weights, out_dtype="int8")
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params


def create_model_broken():
    # initial setup
    input_shape = (1,3,32,32)
    params = {}
    input_tensor = relay.var("input", relay.TensorType(input_shape, 'int8'))

    # Connect resnet blocks
    repeat_in_block = 2
    repeat_block = 3
    x, params_out = relay_soma_conv2d(input_tensor,
                "conv1",
                (16,3,3,3), np.ones((16,3,3,3), dtype=np.int8),
                np.ones(16, dtype=np.int32))
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_block_16", 16, 16,
                                    repeat_in_block, repeat_block)
    params.update(params_out)
    ## here is the error
    #x, params_out = relay_res_layer(x, "res_block_32", 16, 32,
    #                                repeat_in_block, repeat_block)
    #params.update(params_out)
    #x, params_out = relay_res_layer(x, "res_block_64", 32, 64,
    #                                repeat_in_block, repeat_block,
    #                                stride_in_last=False)
    #params.update(params_out)

    #x = relay.nn.avg_pool2d(x, (8,8))
    #x = relay.reshape(x, (1,64))

    #fc_weights_name = "fc_weights"
    #fc_weights_shape = (10,64)
    #fc_weights = relay.var(fc_weights_name, 
    #                       relay.TensorType(fc_weights_shape, "int8"))
    #params.update({fc_weights_name: tvm.nd.array(np.ones(fc_weights_shape, 
    #                                                      dtype=np.int8))})
    #x = relay.nn.dense(x, fc_weights, out_dtype="int8")
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params


if __name__ == "__main__":
    target, device, measurement, interactive, fusion, gcc_opt = parse_cli_options()
    mod, params = create_model()
    model = TVMCModel(mod, params)
    # compile the model
    tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
    create_demo_file(mod)
    fusion_name = "fused" if fusion else "unfused"
    target_name = "dory" if target == "soma_dory, c" else "c"
    csv_name = f"relay_resnet20_{target_name}_{fusion_name}" + \
               f"_O{gcc_opt}_{measurement}.csv"
    #insert_profiler(measurement=measurement,
    #                interactive=interactive,
    #                csv_file=csv_name)
    make(device)
    
