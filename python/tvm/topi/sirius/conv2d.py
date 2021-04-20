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
from tvm import te  # Used to manipulate FTVMSchedules and FTVMComputes
from ..utils import traverse_inline
import logging
import tvm
from tvm import topi
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad
################################### CLASSES ####################################

# Your classes go here 

################################## FUNCTIONS ###################################
def intrin_pad(data_shape, filter_shape, padding="SAME", dilation=1, out_dtype="int8", data_layout="NCHW"):

    # Hardcoding the values here
    # Data Layout is expected to be NCHW or NHWC
    # Dimensions of tensor are not variable here as this doesn't allow for proper tensorization of the padding function

    assert data_layout == "NCHW" or data_layout == "NHWC", "Data layout other than NCHW or NHWC not supported"

    dilation_h = dilation_w = dilation
    padding = "SAME"
    if data_layout == "NCHW":
        _, in_channel, in_height, in_width = data_shape
        num_filter, channel, kernel_h, kernel_w = filter_shape
    else:  # data_layout == "NHWC"
        _, in_height, in_width, in_channel = data_shape
        kernel_h, kernel_w, num_filter, channel = filter_shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    input_data = te.placeholder(data_shape, dtype=out_dtype, name="data")

    if data_layout == "NCHW":
        pad_before = [0, 0, pad_top, pad_left]
        pad_after = [0, 0, pad_down, pad_right]
    else:  # data_layout == "NHWC":
        pad_before = [0, pad_top, pad_left, 0]
        pad_after = [0, pad_down, pad_right, 0]
    # Only do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    temp = pad(input_data, pad_before, pad_after, name="pad_temp")

    # Since no splitting is applied here, the data layout strides are inferred automatically
    input_data_buffer = tvm.tir.decl_buffer(input_data.shape, input_data.dtype, name="padded_data_b", offset_factor=1)
    output_data_buffer = tvm.tir.decl_buffer(temp.shape, temp.dtype, name="conv_soma_b", offset_factor=1)

    def intrin_func(ins,outs):
        ib = tvm.tir.ir_builder.create()
        unpadded_in = ins
        padded_out = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                # This is an empty function that does nothing :o
                "int32",
                # TODO This padder is actually not padding correctly
                "soma_wrapped_padder"
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(temp.op, intrin_func, binds={input_data: input_data_buffer,
                                                              temp: output_data_buffer})


def intrin_conv2d_hwlib(data_shape, filter_shape, stride=1, padding="SAME", dilation=1, in_dtype="int8", out_dtype="int32", data_layout="NCHW"):
    # This is essentially a copy of topi.nn.conv2d()
    #
    # Differences:
    #  * Memory layout is fixed.
    #       1)  Kernel = OIHW and Data = NCHW
    #       2)  Kernel = HWOI and Data = NHWC
    #  * dilation of 1 is hardcoded
    #  * stride of 1 is hardcoded
    #  * Padding is hardcoded so the output has the same dimensions as the input
    #  * intrinsic expects no batch dimension, so batch dimension can be unrolled by tensorization.
    #  * TVM does not allow for nested intrinsic definition! This intrinsic assumes the input is already padded!

    assert data_layout == "NCHW" or data_layout == "NHWC", "Data layout other than NCHW or NHWC not supported"

    # Hardcoding the values here
    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    padding = "SAME"

    if data_layout == "NCHW":
        _, in_channel, in_height, in_width = data_shape
        num_filter, channel, kernel_h, kernel_w = filter_shape
    else:  # data_layout == "NHWC":
        _, in_height, in_width, in_channel = data_shape
        kernel_h, kernel_w, num_filter, channel = filter_shape

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    if data_layout == "NCHW":
        padded_input_shp = (1, in_channel, in_height+pad_top+pad_down, in_width+pad_left+pad_right,)
    else: # data_layout == "NHWC":
        padded_input_shp = (1, in_height+pad_top+pad_down, in_width+pad_left+pad_right, in_channel)

    # Assume data is already padded
    # input_data = te.placeholder(input_shp, dtype=data_type, name="data")
    filter_data = te.placeholder(filter_shape, dtype=in_dtype, name="kernel")

    out_channel = num_filter
    out_height = tvm.topi.nn.utils.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = tvm.topi.nn.utils.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # Don't do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    # temp =  pad(input_data, pad_before, pad_after, name="pad_temp")
    padded = te.placeholder(padded_input_shp, dtype=in_dtype, name="padded_data")

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    if data_layout == "NCHW":
        conv_soma = te.compute(
            (out_channel, out_height, out_width),
            lambda ff, yy, xx: te.sum(
                padded[0, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                    out_dtype
                )
                * filter_data[ff, rc, ry, rx].astype(out_dtype),
                axis=[rc, ry, rx],
                ),
            tag="conv2d_nchw",
        )
    else: # data_layout == "NHWC":
        conv_soma = te.compute(
            (out_height, out_width, out_channel),
            lambda yy, xx, ff: te.sum(
                padded[0, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc].astype(
                    out_dtype
                )
                * filter_data[ry, rx, ff, rc].astype(out_dtype),
                axis=[ry, rx, rc],
                ),
            tag="conv2d_nhwc",

        )

    # Since no splitting is applied here, the data layout strides are inferred automatically
    padded_buffer = tvm.tir.decl_buffer(padded.shape, padded.dtype, name="padded_data_b", offset_factor=1)
    filter_data_buffer = tvm.tir.decl_buffer(filter_data.shape, filter_data.dtype, name="kernel_b", offset_factor=1)
    output_data_buffer = tvm.tir.decl_buffer(conv_soma.shape, conv_soma.dtype, name="conv_soma_b", offset_factor=1)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        data_in, kernel_in = ins
        conv_out = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                # TODO this call is actually only really valid for NCHW; not NHWC
                "int32",
                "soma_wrapped_conv2d",
                data_in.access_ptr("r"),
                kernel_in.access_ptr("r"),
                conv_out.access_ptr("w"),
                in_width,       # uint32_t w
                in_height,      # uint32_t h
                in_channel,     # uint32_t c
                kernel_w,       # uint32_t fx
                kernel_h,       # uint32_t fy
                out_channel,    # uint32_t k
                out_width,      # uint32_t ox
                out_height,     # uint32_t oy
                stride,         # uint32_t stride
                8,              # uint32_t precision
                0,              # uint32_t activation_function
                # (Ox_padded - Ox)*(2**4) --> Ox_padded - Ox illustrates how much "padding" was needed to "fill" the PE array
                # TODO remove hardcoded value for 0
                # pad_left and pad_right should be equal!
                pad_down*(2**12)+ pad_top*(2**8)+ 0 + pad_left,          # uint32_t zero_padding
                0,              # uint32_t shift_fixed_point
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(conv_soma.op, intrin_func, binds={padded: padded_buffer,
                                                                   filter_data: filter_data_buffer,
                                                                   conv_soma: output_data_buffer})

def fallback_schedule_conv2d(outs):
    """
    similar to topi.generic.nn.schedule_conv2d_hwcn
    note: Even though the name suggests otherwise,
    this function just returns the default generic schedule with autoinline set to false
    So it works generally on all con2d operations, regardless of mem layout
    """
    return topi.generic.schedule_conv2d_hwcn(outs)


def schedule_conv2d(outs):
    """Schedule for conv2d, specialized for SIRIUS.

    Supports
    --------
    1) conv2d_nchw = Data : NCHW, Kernel : OIHW
    2) conv2d_nhwc = Data : NHWC, Kernel : HWOI

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d
          in the format of an array of tensors.

    Returns
    -------
    schedule: Schedule
        The computation schedule for the op.
    """
    # We start from TVM's default schedule
    s = te.create_schedule([x.op for x in outs])
    # Debug message
    print("Using SIRIUS schedule")

    def _callback(op):
        if ("conv2d_nchw" in op.tag) or ("conv2d_nhwc" in op.tag):
            _, kernel_orig = op.input_tensors
            data_orig = s[op].op.input_tensors[0].op.input_tensors[0]
            if "conv2d_nchw" in op.tag:
                padding_intrinsic = intrin_pad(data_orig.shape, kernel_orig.shape, data_layout="NCHW")
                conv2d_intrinsic = intrin_conv2d_hwlib(data_orig.shape, kernel_orig.shape, data_layout="NCHW")
            else:  # "conv2d_nhwc" in op.tag:
                padding_intrinsic = intrin_pad(data_orig.shape, kernel_orig.shape, data_layout="NHWC")
                conv2d_intrinsic = intrin_conv2d_hwlib(data_orig.shape, kernel_orig.shape, data_layout="NHWC")
            # Get padded tensor and tensorize padding
            s.stages[1].tensorize(op.input_tensors[0].op.axis[0], padding_intrinsic)
            # Now also tensorize the conv2d operation without the padding
            s.stages[3].tensorize(op.axis[1], conv2d_intrinsic)

    # Traverse the given operation's stages, and tensorize where needed!
    traverse_inline(s, outs[0].op, _callback)
    # Returning the built schedule
    return s

#def schedule_conv2d_nchw(outs):
#    """Schedule for conv2d_nchw, specialized for SIRIUS.
#
#    Parameters
#    ----------
#    outs: Array of Tensor
#          The computation graph description of conv2d_nchw
#          in the format of an array of tensors.
#
#    Returns
#    -------
#    schedule: Schedule
#        The computation schedule for the op.
#    """
#    # We start from TVM's default schedule
#    s = te.create_schedule([x.op for x in outs])
#    # Debug message
#    print("Using SIRIUS schedule")
#
#    def _callback(op):
#        if "conv2d_nchw" in op.tag:
#            _, kernel_orig = op.input_tensors
#            data_orig = s[op].op.input_tensors[0].op.input_tensors[0]
#            # Get padded tensor and tensorize padding
#            s.stages[1].tensorize(op.input_tensors[0].op.axis[0],intrin_pad(data_orig.shape,kernel_orig.shape))
#            # Now also tensorize the conv2d operation without the padding
#            s.stages[3].tensorize(op.axis[1], intrin_conv2d_hwlib(data_orig.shape, kernel_orig.shape))
#
#    # Traverse the given operation's stages, and tensorize where needed!
#    traverse_inline(s, outs[0].op, _callback)
#    # Returning the built schedule
#    return s
##################################### MAIN #####################################

if __name__ == "__main__":
    # The code to run when this file is used as a script goes here
    pass

##################################### EOF ######################################
