# This is a simple example of a CONV2D operation for SOMA
# This example leaves the Data layout preparation step for the HWLib instead of doing it inside TVM


import tvm
from tvm import te
from tvm import topi
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad

from tvm.contrib import tedd
# Tensor Dimensions

B = 2       # Input batch (also called N sometimes)
C = 3       # Input channels
X = 256     # Input width (also called W sometimes)
Y = 256     # Input height (also called H sometimes)

FX = 5      # Filter width
FY = 5      # Filter height
K = 32      # Filter output channels

data_type = "int8"

def intrin_pad(data_shape, filter_shape, stride=1, padding="SAME", dilation=1, out_dtype="int8"):

    # Hardcoding the values here
    # Dimensions of tensor are not variable here as this doesn't allow for proper tensorization of the padding function

    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    padding = "SAME"
    input_shp = data_shape
    filter_shp = filter_shape
    _, in_channel, in_height, in_width = input_shp
    num_filter, channel, kernel_h, kernel_w = filter_shp


    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    input_data = te.placeholder(input_shp, dtype=data_type, name="data")

    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    # Only do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    temp =  pad(input_data, pad_before, pad_after, name="pad_temp")


    print("Preview of tensorized intrinsic:")
    print("================================")
    preview = te.create_schedule(temp.op)
    print(tvm.lower(preview, [input_data], simple_mode=True))

    # Since no splitting is applied here, the data layout strides are inferred automatically
    input_data_buffer = tvm.tir.decl_buffer(input_data.shape, input_data.dtype, name="padded_data_b", offset_factor=1)
    output_data_buffer = tvm.tir.decl_buffer(temp.shape, temp.dtype, name="conv_soma_b", offset_factor = 1)

    def intrin_func(ins,outs):
        ib = tvm.tir.ir_builder.create()
        unpadded_in = ins
        padded_out = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                # This is an empty function that does nothing :o
                "int32",
                "soma_wrapped_padder"
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(temp.op, intrin_func, binds={input_data: input_data_buffer,
                                                                   temp: output_data_buffer})


def intrin_conv2d_hwlib(stride=1, padding="SAME", dilation=1, out_dtype="int8"):
    # This is essentially a copy of topi.nn.conv2d()
    #
    # Differences:
    #  * dilation of 1 is hardcoded
    #  * stride of 1 is hardcoded
    #  * Padding is hardcoded so the output has the same dimensions as the input
    #  * intrinsic expects no batch dimension, so batch dimension can be unrolled by tensorization.
    #  * TVM does not allow for nested intrinsic definition! This intrinsic assumes the input is already padded!

    c = te.var(name="c")
    x = te.var(name="x")
    y = te.var(name="y")

    fx = te.var(name="fx")
    fy = te.var(name="fy")
    k = te.var(name="k")

    # Hardcoding the values here
    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    padding = "SAME"
    input_shp = (1, c, y, x)
    filter_shp = (k,c,fy,fx)
    _, in_channel, in_height, in_width = input_shp
    num_filter, channel, kernel_h, kernel_w = filter_shp


    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    padded_input_shp = (1,c,y+pad_top+pad_down,x+pad_left+pad_right)

    #input_data = te.placeholder(input_shp, dtype=data_type, name="data")
    filter_data = te.placeholder(filter_shp, dtype=data_type, name="kernel")

    out_channel = num_filter
    out_height = tvm.topi.nn.utils.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = tvm.topi.nn.utils.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    print(pad_top)
    print(pad_left)
    print(pad_down)
    print(pad_right)
    # Don't do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    #temp =  pad(input_data, pad_before, pad_after, name="pad_temp")
    padded = te.placeholder(padded_input_shp, dtype=out_dtype, name="padded_data")

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

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
    print("Preview of tensorized intrinsic:")
    print("================================")
    preview = te.create_schedule(conv_soma.op)
    print(tvm.lower(preview, [padded, filter_data], simple_mode=True))

    # Since no splitting is applied here, the data layout strides are inferred automatically
    padded_buffer = tvm.tir.decl_buffer(padded.shape, padded.dtype, name="padded_data_b", offset_factor=1)
    filter_data_buffer = tvm.tir.decl_buffer(filter_data.shape, filter_data.dtype, name="kernel_b", offset_factor=1)
    output_data_buffer = tvm.tir.decl_buffer(conv_soma.shape, conv_soma.dtype, name="conv_soma_b", offset_factor = 1)

    def intrin_func(ins,outs):
        ib = tvm.tir.ir_builder.create()
        data_in, kernel_in = ins
        conv_out = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "soma_wrapped_conv2d",
                data_in.access_ptr("r"),
                kernel_in.access_ptr("r"),
                conv_out.access_ptr("w"),
                x,              # uint32_t w
                y,              # uint32_t h
                c,              # uint32_t c
                fx,             # uint32_t fx
                fy,             # uint32_t fy
                k,              # uint32_t k
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

data_orig = te.placeholder((B,C,Y,X),dtype=data_type, name="data_orig")
kernel_orig = te.placeholder((K,C,FY,FX), dtype=data_type, name="kernel_orig")
conv2d = topi.nn.conv2d(data_orig,kernel_orig,1,"SAME",1)


print("Generic Schedule for Element-wise Sum Compute to be tensorized:")
print("===============================================================")
schedule = te.create_schedule(conv2d.op)
print(tvm.lower(schedule, [data_orig, kernel_orig], simple_mode=True))

# Get padded tensor and tensorize padding
schedule.stages[1].tensorize(schedule[conv2d].op.input_tensors[0].op.axis[0],intrin_pad(data_orig.shape, kernel_orig.shape))
print(tvm.lower(schedule, [data_orig, kernel_orig], simple_mode=True))
# Now also tensorize the conv2d operation without the padding
schedule[conv2d].tensorize(schedule[conv2d].op.axis[1],intrin_conv2d_hwlib())
print(tvm.lower(schedule, [data_orig, kernel_orig], simple_mode=True))

