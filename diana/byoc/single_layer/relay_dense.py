import tvm
import tvm.relay as relay
import numpy as np
import utils
from profiler import insert_profiler
from tvm.driver.tvmc.model import TVMCModel
from typing import Tuple, Optional
from numpy import typing as npt

def create_model(weight_bits: int,
                 act: bool = True,
                 input_shape: Tuple[int,...] = (1, 16),
                 weights_shape: Tuple[int,...] = (128, 16),
                 weights_values: Optional[tvm.nd.array] = None,
                 bias_values: Optional[tvm.nd.array] = None,
                 shift_bits: int = 4
                 ):
    """
    Generate a small relay graph that performs a DIANA-accelerator-
    eligible dense pattern with various parameters.

    Note: that dense is currently only supported on the digital core
    """
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))
    if weights_values is None:
        weights = utils.create_random_array(weights_shape, 
                                            f'int{weight_bits}')
    else:
        weights = weights_values
    if bias_values is None:
        bias = utils.create_random_array(weights_shape[0], 'int32')
    else:
        bias = bias_values
    x, params1 = utils.relay_soma_dense(x, 'dense1', weights, bias, 
                                         act=act, 
                                         shift_bits=shift_bits)
    params = params1
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    return mod, params


if __name__ == "__main__":
    # for reproducability
    np.random.seed(0)
    # Get options from cli
    args, opt_string = utils.parse_cli_options()
    # create the model
    mod, params = create_model(args.weight_bits)
    model = TVMCModel(mod, params)
    # compile the model
    utils.tvmc_compile_and_unpack(model, target=args.target, 
                                  fuse_layers=args.fusion)
    indefinite = True if args.measurement == "power" else False
    utils.create_demo_file(mod, indefinite=indefinite)
    insert_profiler(measurement=args.measurement,
                    interactive=args.interactive,
                    csv_file="relay_dense_"+opt_string+".csv")
    if not args.interactive:
        utils.make(args.device)
        utils.gdb(args.device)
