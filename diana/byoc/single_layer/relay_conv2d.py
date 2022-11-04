import utils
from profiler import insert_profiler
import tvm
import tvm.relay as relay
from tvm.driver.tvmc.model import TVMCModel
import numpy as np

def create_model(weight_bits):
    random = True
    input_shape = (1, 32, 32, 32)
    weights_shape = (32, 32, 3, 3)
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))
    if random:
        weights = utils.create_random_array(weights_shape, f'int{weight_bits}')
        bias = utils.create_random_array(weights_shape[0], 'int32')
    else:
        weights = utils.numpy_to_array(np.ones(weights_shape,dtype="int8"), f'int{weight_bits}')
        bias = utils.numpy_to_array(np.zeros(weights_shape[0], dtype="int32"), 'int32')
    x, params1 = utils.relay_soma_conv2d(x, 'conv1', weights, bias, 
                                         padding=(1, 1), act=False, 
                                         shift_bits=4)
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
                    csv_file="relay_conv2d_"+opt_string+".csv")
    if not args.interactive:
        utils.make(args.device)
        utils.gdb(args.device)
