import utils
from profiler import insert_profiler
import tvm
import tvm.relay as relay
from tvm.driver.tvmc.model import TVMCModel
import numpy as np

def create_model(weight_bits):
    raise NotImplementedError

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
