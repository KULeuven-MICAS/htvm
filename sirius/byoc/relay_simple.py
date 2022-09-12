from utils import (
        tvmc_compile_and_unpack, 
        relay_soma_conv2d,
        create_demo_file, 
        parse_cli_options,
        load_or_create_random_array
        )
from benchmark import create_benchmark
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import numpy as np


def create_model():
    input_shape = (1, 3, 16, 16)
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    weights_shape = (32, 3, 3, 3)
    #special_data = np.array([[-7,-5,-3,-2,-1,0,1,2,3] for i in range(16*3)])
    #special_data = special_data.reshape(weights_shape).astype(np.int8)
    special_data = load_or_create_random_array("weights.npy",
                                               weights_shape, np.int8)
    x, params1 = relay_soma_conv2d(x, 'conv1', weights_shape, 
                                   special_data,
                                   np.ones(weights_shape[0]).astype(np.int32), 
                                   act=False, shift_bits=4)
   # weights_shape = (32, 16, 3, 3)
   # special_data = np.array([[-1,0,1] for i in range(32*16*3)])
   # special_data = special_data.reshape(weights_shape).astype(np.int8)
   # x, params2 = relay_soma_conv2d(x, 'conv2', weights_shape,
   #                                special_data, 
   #                                np.ones(weights_shape[0]).astype(np.int32), 
   #                                act=True, shift_bits=5)

   # weights_shape = (16, 32, 3, 3)
   # special_data = np.array([[-5,-4,-3,-2,-1,0,1,2,3] for i in range(32*16)])
   # special_data = special_data.reshape(weights_shape).astype(np.int8)
   # x, params3 = relay_soma_conv2d(x, 'conv3', weights_shape,
   #                                special_data,
   #                                np.ones(weights_shape[0]).astype(np.int32),
   #                                strides=(2,2),
   #                                act=False, shift_bits=3)

   # y_shape = (1, 16, 16, 16)
   # y_name = "input_y"
   # y = relay.var(y_name, relay.TensorType(y_shape, 'int8'))
   # y_value = np.ones(y_shape).astype(np.int8)
   # y_param = {y_name: tvm.nd.array(y_value)} 

   # x = relay.add(x, y)

   # # combine all params in one dictionary
   # params1.update(params2)
   # params1.update(params3)
   # params1.update(y_param)
    params = params1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    return mod, params


if __name__ == "__main__":
    target, measurement, interactive, fusion, gcc_opt = parse_cli_options()
    # create the model
    mod, params = create_model()
    model = TVMCModel(mod, params)
    # compile the model
    tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
    create_demo_file(mod)
    fusion_name = "fused" if fusion else "unfused"
    target_name = "dory" if target == "soma_dory, c" else "c"
    csv_name = f"relay_simple_{target_name}_{fusion_name}_O{gcc_opt}.csv"
    create_benchmark(measurement=measurement,
                    interactive=interactive,
                    csv_file=csv_name)
