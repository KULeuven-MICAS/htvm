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


@conv2d_strategy.register(["cpu"])
def conv2d_strategy_sirius(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    data, kernel = inputs

    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    """
    # These prints are useful for debugging
    print(f"inputs:{inputs}")
    print(f"dilations h:{dilation_h} w:{dilation_w}")
    print(f"strides h:{stride_h} w:{stride_w}")
    print(f"data layout = {layout}")
    print(f"kernel layout = {kernel_layout}")
    print(f"padding = {padding}")
    print(f"groups = {groups}")
    """

    """
    Test if tensorization is applicable. Otherwise use default strategy
    
    Tensorization is applicable for:
    * dilation = 1
    * stride = 1
    * dtype = int8
    * kernel_layout = OIHW
    * data_layout = NCHW
    """

    # Make sure padding is matched to kernel dimensions so that output tensor has same size
    # padding tuple can be specified in multiple ways
    if len(padding) == 4:
        test_same_h_padding = padding[0] == padding[2] == kernel.shape[2]//2
        test_same_w_padding = padding[1] == padding[3] == kernel.shape[3]//2
    else: #len(padding) == 2:
        test_same_h_padding = padding[0] == kernel.shape[2]//2
        test_same_w_padding = padding[1] == kernel.shape[3]//2

    if (data.dtype != "int8") and (kernel.dtype != "int8"):
        return fallback_default_conv2d(strategy)
    if not (test_same_w_padding and test_same_h_padding):
        return fallback_default_conv2d(strategy)
    if layout == "NCHW":
        if kernel_layout == "OIHW":
            logger.warning("SIRIUS Tensorization approach")
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nchw),
                wrap_topi_schedule(topi.sirius.schedule_conv2d_nchw)
            )
        else:
            return fallback_default_conv2d(strategy)
    else:
        return fallback_default_conv2d(strategy)
    # When tensorization is possible return this strategy
    return strategy

def fallback_default_conv2d(strategy):
    logger.warning("SIRIUS conv2d: operation not supported: using fallback")
    strategy.add_implementation(
        #wrap_compute_conv2d(topi.nn.conv2d, need_data_layout=True),
        #wrap_topi_schedule(topi.sirius.fallback_schedule_conv2d)
        #wrap_compute_conv2d(topi.x86.conv2d_nchw),
        #wrap_topi_schedule(topi.x86.schedule_conv2d_nchw)
        wrap_compute_conv2d(topi.arm_cpu.conv2d_nhwc_spatial_pack),
        wrap_topi_schedule(topi.arm_cpu.schedule_conv2d_nhwc_spatial_pack)
    )
    return strategy
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

"""
@schedule_injective.register(["cpu"], override=True)
# Will fail for cpu target if override is not set to True (Default=False)
def schedule_injective_sirius(_, outs, target):
    with target:
        return topi.sirius.schedule_injective(outs)
        """


if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass
