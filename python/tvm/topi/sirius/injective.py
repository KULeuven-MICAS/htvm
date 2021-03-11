#!/usr/bin/env python3

################################### METADATA ###################################

# Contributors: Vincent Tableau Roche
# Contacts: vincent.tableau@esat.kuleuven.be
# Creation Date: 2021-03-04
# Language: Python3

################################### IMPORTS ####################################

# Standard library 
# Your imports from the standard library go here 


# External imports 
# Your imports from other packages go here 


# Internal imports
import tvm
from tvm import te  # Used for schedule manipulations
from ..utils import is_empty_shape # Used for schedule_injective

import logging
logger = logging.getLogger("strategy")

################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################

def schedule_injective_from_existing(sch, out):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """
    if len(sch[out].op.axis) >= 4:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 3:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 2:
        sch[out].parallel(sch[out].op.axis[0])
    return sch


def intrin_ews(ro,co,data_type,stride):
    a = te.placeholder((ro,co), dtype=data_type, name="a")
    b = te.placeholder((ro,co), dtype=data_type, name="b")
    c = te.compute((ro,co), lambda i,j: a[i,j] + b[i,j], name="c")

    # Define buffers
    # Offset factor --> optimize for vectorized buffering
    # Strides are set by the factors that appear near the i.inner and j.inner
    # In this case i.inner corresponds to the columnn dimension of the tensor, so:
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[stride,1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[stride,1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[stride,1])

    def intrin_func(ins, outs):
        # create IR builder
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "soma_wrapped_ews",
                aa.access_ptr("r"), # "r" Results in a "1" in the 5th access pointer field
                bb.access_ptr("r"),
                cc.access_ptr("w"), # "w" Results in a "2" in the 5th access pointer field
                a.shape[0], # width
                a.shape[1], # height
                1,          # channels
                8,          # precision
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})

def intrin_ews_soma(width, data_type, stride_outermost, stride_innermost):
    # Make height and channels variable
    height = te.var(name="height")
    channels = te.var(name="channels")
    tensor_size = (width, height, channels)

    a = te.placeholder(tensor_size, dtype=data_type, name="a")
    b = te.placeholder(tensor_size, dtype=data_type, name="b")

    c = te.compute(tensor_size, lambda i, j, k: a[i, j, k] + b[i, j, k], name="c")

    # Preview a generic schedule
    print("Generic Schedule for Element-wise Sum Compute to be tensorized:")
    print("===============================================================")
    preview = te.create_schedule(c.op)
    print(tvm.lower(preview, [a, b, c], simple_mode=True))

    # Define buffers
    # Offset factor --> optimize for vectorized buffering
    # Strides are set by the factors that appear near the indexing elements
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[stride_outermost, stride_innermost,1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[stride_outermost, stride_innermost,1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[stride_outermost, stride_innermost,1])

    def intrin_func(ins, outs):
        # create IR builder
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "soma_wrapped_ews",
                aa.access_ptr("r"),  # "r" Results in a "1" in the 5th access pointer field
                bb.access_ptr("r"),
                cc.access_ptr("w"),  # "w" Results in a "2" in the 5th access pointer field
                a.shape[0],  # reads out the width value (fixed)
                a.shape[1],  # reads out the height value which is variable!
                a.shape[2],  # reads out the channels value which is variable!
                8,  # precision
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def schedule_injective(outs):
    """SIRIUS platform schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    x = outs[0]

    element_wise_size = 2

    if outs.dim == 2:  # Use the old method
        xo, yo, xi, yi = s[x].tile(x.op.axis[-2],
                                   x.op.axis[-1],
                                   x_factor=element_wise_size,
                                   y_factor=element_wise_size)
        stride = s[x].op.axis[-1].dom.extent
        s[x].tensorize(xi, intrin_ews(element_wise_size,
                                      element_wise_size,
                                      x.dtype,
                                      stride=stride))
        return s

    else:  # Use the new method
        yo, yi = s[x].split(x.op.axis[-3], factor=element_wise_size)
        stride_innermost = s[x].op.axis[-1].dom.extent
        stride_outermost = s[x].op.axis[-2].dom.extent * s[x].op.axis[-1].dom.extent
        s[x].tensorize(yi, intrin_ews_soma(element_wise_size,
                                           x.dtype,
                                           stride_outermost=stride_outermost,
                                           stride_innermost=stride_innermost))
        return s
    #if list(s[x].op.axis):
    #    # do not vectorize for broadcast
    #    (io, ii) = s[x].split(list(s[x].op.axis)[-1], 4)
    #    s[x].vectorize(ii)
    #te.schedule.AutoInlineInjective(s)

    #if not is_empty_shape(x.shape):
    #    schedule_injective_from_existing(s, x)
    #return s

##################################### MAIN #####################################

if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass

##################################### EOF ######################################
