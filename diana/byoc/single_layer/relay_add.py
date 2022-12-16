import utils
from profiler import insert_profiler
import tvm
import tvm.relay as relay
from tvm.driver.tvmc.model import TVMCModel
import numpy as np
from typing import Tuple, Optional
from numpy import typing as npt


def create_model(act: bool = False,
                 shift_bits: int = 4,
                 input_shape: Tuple[int, ...] = (1,2,3,4)):
    """
    Generate a small relay graph that performs a DIANA-accelerator-
    eligible addition pattern with various parameters
    """
    x = relay.var("input_x", relay.TensorType(input_shape, 'int8'))
    y = relay.var("input_y", relay.TensorType(input_shape, 'int8'))
    out = utils.relay_soma_add(x, y, "add_1", act = act, 
                               shift_bits = shift_bits)
    params = {}
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(out)
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
                    csv_file="relay_add_"+opt_string+".csv")
    if not args.interactive:
        utils.make(args.device)
        utils.gdb(args.device)
