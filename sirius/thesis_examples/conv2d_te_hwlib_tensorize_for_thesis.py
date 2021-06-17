# This is a simple example of a CONV2D operation for SOMA
# This example leaves the Data layout preparation step for the HWLib instead of doing it inside TVM


import tvm
from tvm import te
from tvm import topi
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.nn.pad import pad

# Tensor Dimensions

B = 2       # Input batch (also called N sometimes)
C = 3       # Input channels
X = 256     # Input width (also called W sometimes)
Y = 256     # Input height (also called H sometimes)

FX = 5      # Filter width
FY = 5      # Filter height
K = 32      # Filter output channels

data_type = "int8"

def intrin_pad(input_shp, filter_shp, padding="SAME", dilation=1, out_dtype="int8"):

    # Hardcoding the values here
    # Data layout is expected to be NCHW
    # Dimensions of tensor are not variable here as this doesn't allow for proper tensorization of the padding function

    dilation_h = dilation_w = dilation
    _, in_channel, in_height, in_width = input_shp
    num_filter, channel, kernel_h, kernel_w = filter_shp
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    input_data = te.placeholder(input_shp, dtype=out_dtype, name="data")

    # compute graph
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    # Only do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    temp = pad(input_data, pad_before, pad_after, name="pad_temp")
#   print("Preview of tensorized intrinsic:")
#   print("================================")
#   preview = te.create_schedule(temp.op)
#   print(tvm.lower(preview, [input_data], simple_mode=True))

    # Since no splitting is applied here, the data layout strides are inferred automatically
    input_data_buffer = tvm.tir.decl_buffer(input_data.shape, input_data.dtype,
                                            name="padded_data_b", offset_factor=1)
    output_data_buffer = tvm.tir.decl_buffer(temp.shape, temp.dtype,
                                             name="conv_acc_b", offset_factor=1)
    def intrin_func(ins,outs):
        ib = tvm.tir.ir_builder.create()
        unpadded_in = ins[0]
        padded_out = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "acc_padder",
                unpadded_in.access_ptr("r"),
                padded_out.access_ptr("w"),
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(temp.op, intrin_func,
                                 binds={input_data: input_data_buffer,
                                        temp: output_data_buffer})


def intrin_conv2d(input_shp, filter_shp, stride=1, padding="SAME", dilation=1, out_dtype="int8"):
    # This is essentially a copy of topi.nn.conv2d()
    #
    # Differences:
    #  * Memory layout is fixed: Kernel = OIHW , Data = NCHW
    #  * dilation of 1 is hardcoded
    #  * stride of 1 is hardcoded
    #  * Padding is hardcoded so the output has the same dimensions as the input
    #  * intrinsic expects no batch dimension, so batch dimension can be unrolled by tensorization.
    #  * TVM does not allow for nested intrinsic definition! This intrinsic assumes the input is already padded!

    # Hardcoding the values here
    data_type = out_dtype
    stride_h = stride_w = stride
    dilation_h = dilation_w = dilation
    batch, in_channel, in_height, in_width = input_shp
    out_channel, channel, kernel_h, kernel_w = filter_shp

    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    padded_input_shp = (batch, in_channel, in_height+pad_top+pad_down, in_width+pad_left+pad_right)
    # Don't do padding here! TVM does not allow defining an intrinsic for nested compute ops :(
    padded = te.placeholder(padded_input_shp, dtype=out_dtype, name="padded_data")
    filter_data = te.placeholder(filter_shp, dtype=data_type, name="kernel")
    out_height = tvm.topi.nn.utils.simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = tvm.topi.nn.utils.simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    conv_soma = te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            padded[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                out_dtype
            )
            * filter_data[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
            ),
        tag="conv2d_nchw",
    )
#    print("Preview of tensorized intrinsic:")
#    print("================================")
#    preview = te.create_schedule(conv_soma.op)
#    print(tvm.lower(preview, [padded, filter_data], simple_mode=True))

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
                "acc_conv2d",
                data_in.access_ptr("r"),
                kernel_in.access_ptr("r"),
                conv_out.access_ptr("w")
        ))
        return ib.get()
    return te.decl_tensor_intrin(conv_soma.op, intrin_func, binds={padded: padded_buffer,
                                                                   filter_data: filter_data_buffer,
                                                                   conv_soma: output_data_buffer})

# Tensor Dimensions
n = 2       # Input batch
c = 3       # Input channels
h = 256     # Input height
w = 256     # Input width
o = 32      # Filter output channels
i = c       # Filter input channels
fh = 5      # Filter height
fw = 5      # Filter width
data_type = "int8"
data = te.placeholder((n, c, h, w),dtype=data_type, name="data")
kernel = te.placeholder((o, i, fh, fw), dtype=data_type, name="kernel")
conv2d = topi.nn.conv2d(data, kernel, 1, "SAME", 1)
schedule = te.create_schedule(conv2d.op)
schedule.stages[1].tensorize(
    schedule[conv2d].op.input_tensors[0].op.axis[0],
    intrin_pad(data.shape, kernel.shape)
)
schedule[conv2d].tensorize(schedule[conv2d].op.axis[0],
                           intrin_conv2d(data.shape, kernel.shape)
)
print(tvm.lower(schedule,[data, kernel], simple_mode=True))
