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
    input_shape = (1, 3, 16, 16)
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    weights_shape = (32, 3, 3, 3)
    weights = create_random_array(weights_shape, f'int{weight_bits}')
    bias = create_random_array(weights_shape[0], 'int32')
    x, params1 = relay_soma_conv2d(x, 'conv1', weights, bias, padding=(1, 1), act=True, shift_bits=4)

    #weights_shape = (16, 32, 3, 3)
    #weights = create_random_array(weights_shape, f'int{weight_bits}')
    #bias = create_random_array(weights_shape[0], 'int32')
    #x, params2 = relay_soma_conv2d(x, 'conv2', weights, bias, padding=(1, 1), act=True, shift_bits=5)

    #weights_shape = (32, 16, 3, 3)
    #weights = create_random_array(weights_shape, f'int{weight_bits}')
    #bias = create_random_array(weights_shape[0], 'int32')
    #x, params3 = relay_soma_conv2d(x, 'conv3', weights, bias, padding=(1, 1), act=False, shift_bits=3)

    ## combine all params in one dictionary
    #params1.update(params2)
    #params1.update(params3)
    params = params1

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
    create_demo_file(mod)
    fusion_name = "fused" if fusion else "unfused"
    target_name = "dory" if target == "soma_dory, c" else "c"
    csv_name = f"relay_simple_{target_name}_{fusion_name}" + \
               f"_O{gcc_opt}_{measurement}.csv"
    insert_profiler(measurement=measurement,
                    interactive=interactive,
                    csv_file=csv_name)
