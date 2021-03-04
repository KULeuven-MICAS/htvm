#!/usr/bin/env python3

################################### METADATA ###################################

# Contributors: Vincent Tableau Roche
# Contacts: vincent.tableau@esat.kuleuven.be
# Creation Date: 2021-03-01
# Language: Python3

################################### IMPORTS ####################################

# Standard library 
import logging
logger = logging.getLogger("strategy")

# External imports 
# Your imports from other packages go here 


# Internal imports 
from tvm import topi, _ffi, te, ir
from tvm.topi.utils import get_const_int, get_const_float, get_const_tuple, get_float_tuple
from tvm.target import generic_func, override_native_generic_func
from .generic import *
from .. import op as _op

################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################

# conv2d_NCHWc
@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy_sirius(attrs, inputs, out_type, target):
    """conv2d_NCHWc generic strategy

    attrs:
        relay.attrs.Conv2DAttrs(0x55def97c1008)
    inputs:
        [Tensor(shape=[1, 64, 7, 7, 8], op.name=placeholder), Tensor(shape=[64, 64, 3, 3, 8, 8], op.name=placeholder)]
    out_type:
        Tensor[(1, 64, 7, 7, 8), float32]
    target:
        sirius -keys=cpu -link-params=0
    """
    print("Using the SIRIUS strategy!")
    print("attrs:");
    print(attrs)
    # attrs fields are described in include/tvm/relay/nn.h
    print("inputs:")
    print(inputs)
    print("out_type:")
    print(out_type)
    print("target:")
    print(target)

    strategy = _op.OpStrategy()

    logger.warning("Using generic implementation for conv2d_NCHWc.generic")
    strategy.add_implementation(
        wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
        wrap_topi_schedule(topi.sirius.schedule_conv2d_nchw),
        name="conv2d_NCHWc.generic",
    )
    return strategy


    # logger.warning("conv2d_NCHWc is not optimized for this platform.")
    # strategy = _op.OpStrategy()
    # if inputs[0].dtype == "int8" or inputs[0].dtype == "uint8":
    #     strategy.add_implementation(
    #         wrap_compute_conv2d(topi.nn.conv2d_NCHWc_int8, True, True),
    #         wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc_int8),
    #         name="conv2d_NCHWc_int8.sirius",
    #     )
    # else:
    #     strategy.add_implementation(
    #         wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
    #         wrap_topi_schedule(topi.generic.schedule_conv2d_NCHWc),
    #         name="conv2d_NCHWc.sirius",
    #     )
    # return strategy


# _op.register_strategy("conv2d_NCHWc", conv2d_NCHWc_strategy_sirius)


##################################### MAIN #####################################

if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass

##################################### EOF ######################################
