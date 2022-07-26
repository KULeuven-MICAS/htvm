from utils import tvmc_compile_and_unpack, relay_soma_conv2d
import tvm
import tvm.relay as relay
import tvm.relay.transform as transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import numpy as np


def create_model():
    input_shape = (1, 3, 32, 32)
    x = relay.var("input", relay.TensorType(input_shape, 'int8'))

    weights_shape = (16, 3, 3, 3)
    x, params1 = relay_soma_conv2d(x, 'conv1', weights_shape, 
                                   np.ones(weights_shape).astype(np.int8), 
                                   np.ones(weights_shape[0]).astype(np.int32), 
                                   False, 4)
    weights_shape = (32, 16, 3, 3)
    x, params2 = relay_soma_conv2d(x, 'conv2', weights_shape,
                                   np.ones(weights_shape).astype(np.int8), 
                                   np.ones(weights_shape[0]).astype(np.int32), 
                                   True, 5)
    weights_shape = (16, 32, 3, 3)
    x, params3 = relay_soma_conv2d(x, 'conv3', weights_shape,
                                   np.ones(weights_shape).astype(np.int8),
                                   np.ones(weights_shape[0]).astype(np.int32),
                                   True, 5)

    y_shape = (1, 16, 32, 32)
    y_name = "input_y"
    y = relay.var(y_name, relay.TensorType(y_shape, 'int8'))
    y_value = np.ones(y_shape).astype(np.int8)
    y_param = {y_name: tvm.nd.array(y_value)} 

    x = relay.add(x, y)

    # combine all params in one dictionary
    params1.update(params2)
    params1.update(params3)
    params1.update(y_param)
    params = params1

    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)

    return mod, params


if __name__ == "__main__":
    # create the model
    mod, params = create_model()
    model = TVMCModel(mod, params)
    # compile the model
    tvmc_compile_and_unpack(model, target="c", fuse_layers=True)
