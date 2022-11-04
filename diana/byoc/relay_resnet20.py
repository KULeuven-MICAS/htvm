import utils
from profiler import insert_profiler
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend import Executor, Runtime
import numpy as np


def relay_res_block(input_tensor, name, input_channels, output_channels,
                    strided=True):
    # initial setup
    params = {}
    # 3x3 convolution block 1
    weights_shape = (output_channels, input_channels, 3, 3)
    weights = utils.create_random_array(weights_shape, dtype="int8")
    bias = utils.create_random_array(output_channels, dtype="int32")
    x, params_out = utils.relay_soma_conv2d(input_tensor, 
                        name + '_3x3conv_0',
                        weights,
                        bias,
                        strides=(1,1),
                        padding=(1,1))
    params.update(params_out)
    # 3x3 convolution block 2
    weights_shape = (output_channels, output_channels, 3, 3)
    weights = utils.create_random_array(weights_shape, dtype="int8")
    bias = utils.create_random_array(output_channels, dtype="int32")
    x, params_out = utils.relay_soma_conv2d(x, 
                        name + '_3x3conv_1',
                        weights, 
                        bias,
                        strides=((2,2) if strided else (1,1)),
                        padding=(1,1))
    params.update(params_out)
    # 1x1 skip layer convolution
    weights_shape = (output_channels,input_channels,1,1)
    weights = utils.create_random_array(weights_shape, dtype="int8")
    bias = utils.create_random_array(output_channels, dtype="int32")
    y, params_out = utils.relay_soma_conv2d(input_tensor, 
                        name + '_1x1conv',
                        weights,
                        bias,
                        strides=((2,2) if strided else (1,1)))
    params.update(params_out)
    x = relay.add(x, y)
    return x, params

def relay_res_layer(input_tensor, name, input_channels, output_channels, strided):
    params = {}
    x, params_out = relay_res_block(input_tensor, name + "block_1",
                                    input_channels,
                                    output_channels, False) 
    params.update(params_out)
    x, params_out = relay_res_block(x, name + "block_2", output_channels,
                                    output_channels, False) 
    params.update(params_out)
    x, params_out = relay_res_block(x, name + "block_3", output_channels,
                                    output_channels, strided) 
    params.update(params_out)
    return x, params

    
def create_model():
    # initial setup
    input_shape = (1,3,32,32)
    params = {}
    input_tensor = relay.var("input", relay.TensorType(input_shape, 'int8'))

    # Connect resnet blocks
    weights = utils.create_random_array((16,3,3,3), dtype="int8")
    bias = utils.create_random_array(16, dtype="int32")
    x, params_out = utils.relay_soma_conv2d(input_tensor, "conv1", weights,
                                      bias, padding=(1, 1))
    params.update(params_out)
    x, params_out = relay_res_layer(x, "res_layer_16", 16, 16, True)
    params.update(params_out)
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
    fc_weights_values = utils.create_random_array(fc_weights_shape, "int8")
    params.update({fc_weights_name: fc_weights_values})
    x = relay.nn.dense(x, fc_weights, out_dtype="int8")
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params



if __name__ == "__main__":
    # for reproducability
    np.random.seed(0)
    # Get options from cli
    args, opt_string = utils.parse_cli_options()
    # create the model
    mod, params = create_model()
    model = TVMCModel(mod, params)
    # compile the model
    utils.tvmc_compile_and_unpack(model, target=args.target, 
                                  fuse_layers=args.fusion)
    indefinite = True if args.measurement == "power" else False
    utils.create_demo_file(mod, indefinite=indefinite)
    insert_profiler(measurement=args.measurement,
                    interactive=args.interactive,
                    csv_file="relay_resnet20_"+opt_string+".csv")
    if not args.interactive:
        utils.make(args.device)
        utils.gdb(args.device)
