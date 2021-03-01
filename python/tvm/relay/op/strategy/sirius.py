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

################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################

# conv2d_NCHWc
@conv2d_NCHWc_strategy.register("cpu")
def conv2d_NCHWc_strategy_sirius(attrs, inputs, out_type, target):
    """conv2d_NCHWc generic strategy"""
    print("Using the SIRIUS strategy!")
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
