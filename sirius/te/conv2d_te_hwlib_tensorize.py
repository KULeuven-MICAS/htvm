# This is a simple example of a CONV2D operation for SOMA
# This example leaves the Data layout preparation step for the


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

def intrin_pad(padding)

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
            padded[1, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
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
                x,      #uint32_t w
                y,      #uint32_t h
                c,      #uint32_t c
                fx,     #uint32_t fx
                fy,     #uint32_t fy
                k,  #uint32_t k
                out_width, #uint32_t ox
                out_height, #uint32_t oy
                stride,      #uint32_t stride
                8,      #uint32_t precision
                0,      #uint32_t activation_function
                0,      #uint32_t zero_padding
                0,      #uint32_t shift_fixed_point
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(conv_soma.op, intrin_func, binds={padded: padded_buffer,
                                                                   filter_data: filter_data_buffer,
                                                                   conv_soma: output_data_buffer})

intrin_conv2d_hwlib()

#

## Generate schedule to apply tensorization to:
#data_orig = te.placeholder((B,C,Y,X),dtype=data_type, name="data_orig")
#kernel_orig = te.placeholder((K,C,FY,FX), dtype=data_type, name="kernel_orig")
## Four dimensional tensor goes in, five dimensional tensor comes out
#kernel_prepared = prepare_filter(kernel_orig, intrinsic_size)
#data_prepared = prepare_data(data_orig, intrinsic_size)
#
#in_batch, in_height, in_width_outer, in_channel, in_width_inner = data_prepared.shape
#num_filter_outer, channel, kernel_h, kernel_w, num_filter_inner = kernel_prepared.shape
#
#rc = te.reduce_axis((0, in_channel), name="rfc")
#rfy = te.reduce_axis((0, kernel_h), name="rfy")
#rfx = te.reduce_axis((0, kernel_w), name="rfx")
#
#conv = te.compute((in_batch,
#                   num_filter_outer,
#                   in_height - kernel_h + 1,
#                   in_width_outer,
#                   in_width_inner - kernel_w + 1,
#                   num_filter_inner,),
#                  lambda b, ko, y, xo, xi, ki: te.sum(data_prepared[in_batch,
#                                                                   y + rfy,
#                                                                   xo,
#                                                                   rc,
#                                                                   xi + rfx].astype(data_type)
#                                                  * kernel_prepared[ko,
#                                                                    rc,
#                                                                    rfy,
#                                                                    rfx,
#                                                                    ki].astype(data_type),
#                                                  axis=[rc, rfy, rfx],),
#                  tag="conv2d_SOMA_to_tensorize",
#                  )
#
#s = te.create_schedule(conv.op)
#
#print("Generic Schedule for Element-wise Sum Compute to be tensorized:")
#print("===============================================================")
#tensorize_me = te.create_schedule(conv.op)
#print(tvm.lower(tensorize_me, [data_orig, kernel_orig], simple_mode=True))
#tensorize_me[conv].tensorize(conv.op.axis[1],intrin_conv2d("int8",intrinsic_size))
#print("Schedule after tensorization")
#print("============================")
#print(tvm.lower(tensorize_me, [data_orig, kernel_orig], simple_mode=True))
#
#tedd.viz_dataflow_graph(tensorize_me, dot_file_path="/tmp/dataflow.dot")
#tedd.viz_schedule_tree(tensorize_me, dot_file_path="/tmp/schedule_tree.dot")
#
#lib = tvm.build(tensorize_me, [data_orig, kernel_orig], target_host="sirius")
#file_name = "conv2d_soma.so"
#lib.export_library(file_name, workspace_dir="/tmp/")