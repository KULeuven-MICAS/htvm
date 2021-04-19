
import tvm
from tvm import te  # Used for schedule manipulations

import logging
logger = logging.getLogger("strategy")


def intrin_ews_soma(width, data_type, stride_outermost, stride_innermost):
    # Make height and channels variable
    height = te.var(name="height")
    channels = te.var(name="channels")
    tensor_size = (width, height, channels)

    a = te.placeholder(tensor_size, dtype=data_type, name="a")
    b = te.placeholder(tensor_size, dtype=data_type, name="b")

    c = te.compute(tensor_size, lambda i, j, k: a[i, j, k] + b[i, j, k], name="c")

    # Define buffers
    # Offset factor --> optimize for vectorized buffering
    # Strides are set by the factors that appear near the indexing elements
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1,
                             strides=[stride_outermost, stride_innermost, 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1,
                             strides=[stride_outermost, stride_innermost, 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1,
                             strides=[stride_outermost, stride_innermost, 1])

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
    # Set the size of the intrinsic
    ELEMENT_WISE_SIZE = 16

    # Test if the outs is a list of multiple values, else make it a list
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    # If you pass a list to create_schedule, it will schedule the operations consecutively
    s = te.create_schedule([x.op for x in outs])
    # Try to tensorize the operators in this schedule
    for x in outs:
        # Check if tensorization is applicable
        if x.op.name != "T_add":
            # No intrinsic available :( ; just use vanilla schedule:
            logger.warning(f'{x.op.name}: No supported intrinsic available')
            continue
        if len(x.shape) < 3:
            # Not enough dimensions to unroll over; just use vanilla schedule:
            # TODO, reshape so this is possible for tensors with too few dimensions?
            logger.warning(f'{x.op.name}: too few tensor dimensions to tensorize: shape = {x.shape}')
            continue
        # Check if all input tensors for add operation are same size
        sizes = []
        for tensor in x.op.input_tensors:
            sizes.append(len(tensor.shape))
        all_equal = True
        for size in sizes:
            # this loop will stop early if one size is encountered which is not the same
            if not all_equal:
                continue
            else:
                all_equal = (sizes[0] == size)
        if not all_equal:
            logger.warning(f'{x.op.name}: Not all tensor sizes are equal: sizes = {sizes}')
            continue
        else:
            # Tensorize!
            logger.warning(f'{x.op.name}: **Tensorizing** element wise sum')
            yo, yi = s[x].split(x.op.axis[-3], factor=ELEMENT_WISE_SIZE)
            stride_innermost = s[x].op.axis[-1].dom.extent
            stride_outermost = s[x].op.axis[-2].dom.extent * s[x].op.axis[-1].dom.extent
            s[x].tensorize(yi, intrin_ews_soma(ELEMENT_WISE_SIZE,
                                               x.dtype,
                                               stride_outermost=stride_outermost,
                                               stride_innermost=stride_innermost))
            continue
    # TODO: Find out why we have to do this :D
    te.schedule.AutoInlineInjective(s)
    return s
