import utils
from profiler import insert_profiler
import tvm
import tvm.relay as relay
from tvm.driver.tvmc.model import TVMCModel
import numpy as np
from typing import Tuple, Optional
from numpy import typing as npt


def create_model(weight_bits: int,
                 act: bool = True,
                 input_shape: Tuple[int, ...] = (1, 32, 32, 32),
                 weights_shape: Tuple[int, ...] = (32, 32, 3, 3),
                 weights_values: Optional[tvm.nd.array] = None,
                 bias_values: Optional[tvm.nd.array] = None,
                 padding: Tuple[int, int] = (1, 1),
                 strides: Tuple[int, int] = (1, 1),
                 shift_bits: int = 4
                 ):
    """
    Generate a small relay graph that performs a DIANA-accelerator-
    eligible convolution pattern with various parameters
    """
    # Using input_0 to be used with create_demo_file
    x = relay.var("input_0", relay.TensorType(input_shape, 'int8'))
    # Get or generate weight_values
    if weights_values is None:
        weights = utils.create_random_array(weights_shape, 
                                            f'int{weight_bits}')
    else:
        weights = weights_values
    # Get or generate bias values
    if bias_values is None:
        bias = utils.create_random_array(weights_shape[0], 'int32')
    else:
        bias = bias_values
    # Generate the conv2d call
    x, params1 = utils.relay_soma_conv2d(x, 'conv1', weights, bias, 
                                         padding=padding, 
                                         strides=strides,
                                         act=act, 
                                         shift_bits=shift_bits)
    params = params1
    # create an IR module from the relay expression
    mod = tvm.ir.IRModule()
    mod = mod.from_expr(x)
    return mod, params
