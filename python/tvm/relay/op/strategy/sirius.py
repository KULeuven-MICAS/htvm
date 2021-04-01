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


@conv2d_strategy.register("cpu")
def conv2d_strategy_sirius(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    logger.warning("Using SIRIUS conv2d strategy")
    data, kernel = inputs
    dilation_h,dilation_w =  attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if padding != "SAME":
        raise NotImplementedError("Padding other than SAME not supported")

    if layout == "NCHW":
        if kernel_layout == "OIHW":
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nchw, True, True),
                wrap_topi_schedule(topi.sirius.schedule_conv2d)
            )
        else:
            raise NotImplementedError(f"Data Layout:{layout}: Kernel layout{kernel_layout} not supported")
    else:
        raise NotImplementedError(f"Data layout{layout} not supported")



# conv2d_NCHWc
#@conv2d_NCHWc_strategy.register("cpu")
#def conv2d_NCHWc_strategy_sirius(attrs, inputs, out_type, target):
#    """conv2d_NCHWc generic strategy
#
#    attrs:
#        relay.attrs.Conv2DAttrs(0x55def97c1008)
#    inputs:
#        [Tensor(shape=[1, 64, 7, 7, 8], op.name=placeholder), Tensor(shape=[64, 64, 3, 3, 8, 8], op.name=placeholder)]
#    out_type:
#        Tensor[(1, 64, 7, 7, 8), float32]
#    target:
#        sirius -keys=cpu -link-params=0
#    """
#    strategy = _op.OpStrategy()
#    logger.warning("Using generic implementation for conv2d_NCHWc.generic")
#    strategy.add_implementation(
#        wrap_compute_conv2d(topi.nn.conv2d_NCHWc, True, True),
#        wrap_topi_schedule(topi.sirius.schedule_conv2d_nchw),
#        name="conv2d_NCHWc.generic",
#    )
#    return strategy


@schedule_injective.register(["cpu"], override=True)
# Will fail for cpu target if override is not set to True (Default=False)
def schedule_injective_sirius(_, outs, target):
    with target:
        return topi.sirius.schedule_injective(outs)


if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass
