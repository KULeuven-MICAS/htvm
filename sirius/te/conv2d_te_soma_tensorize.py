# This is a simple example of a CONV2D operation for SOMA


import tvm
from tvm import te
from tvm import topi


# Tensor Dimensions

B = 2       # Input batch (also called N sometimes)
C = 3       # Input channels
X = 224     # Input width (also called W sometimes)
Y = 224     # Input height (also called H sometimes)

FX = 5      # Filter width
FY = 5      # Filter height
K = 32      # Filter output channels

data_type = "int8"

def prepare_filter(filter, inner_dimension):
    """
    :param filter: 4-D tensor with KCFyFx data layout
    :param inner_dimension: PE array size; fixed inner dimension.
    :return: prepared_filter: 5-D tensor with KoCFyFxKi where Ki is of size `inner_dimension` and Ko*Ki = K
    Note: K is expected to be a multiple of `inner_dimension`
    """
    ki = inner_dimension
    # Calculate the array size and modify length accordingly
    d_shp = filter.shape
    k = int(d_shp[0])
    assert (k % ki == 0), f"K size must be a multiple of {ki}"
    # "split" Kernel dimension K in two pieces Ko,Ki with reshape:
    ko = int(k / ki)
    reshaped = topi.reshape(filter, (ko, ki, d_shp[1], d_shp[2], d_shp[3]))
    # New Kernel layout is Ko,Ki,C,Fx,Fy --> transpose to Ko,C,Fx,Fy,Ki
    return topi.transpose(reshaped, (0, 2, 3, 4, 1))


def prepare_data(data, inner_dimension):
    """
    :param data: 4-D tensor with NCHW aka BCYX data layout
    :param inner_dimension: PE array size; fixed inner dimension
    :return: prepared_data: 5-D tensor with  BYXoCXi layout
    Note: Both Xo and Xi are expected to be a multiple of `inner_dimension`
    """
    xi = inner_dimension
    d_shp = data.shape
    x = int(d_shp[3])
    assert (x % xi == 0), f"X size must be a multiple of {xi}"
    xo = int(x / xi)
    assert (xo % xi == 0), f"Xo size must be a multiple of {xi} too"
    reshaped = topi.reshape(data, (d_shp[0], d_shp[1], d_shp[2], ko, ki))
    # New Data layout is B,C,Y,Xo,Xi --> transpose to B,Y,Xo,C,Xi
    return topi.transpose(reshaped, (0, 2, 3, 1, 4))

def intrin_conv2d(data_type,inner_dim):

    c = te.var(name="c")
    x = te.var(name="x")
    y = te.var(name="y")

    fx = te.var(name="fx")
    fy = te.var(name="fy")
    ko = te.var(name="ko")
    ki = inner_dim

    data = te.placeholder((c, y, x), dtype=data_type, name="data")
    kernel = te.placeholder((ko, c, fy, fx, ki), dtype=data_type, name="kernel")

    rc = te.reduce_axis((0, c), name="rc")
    rfy = te.reduce_axis((0, fy), name="rfy")
    rfx = te.reduce_axis((0, fx), name="rfx")

    conv_soma = te.compute((ko, y-fy+1, x-fx+1, ki),
                           lambda koko, yy, xx, kiki: te.sum(
                               data[rc,
                                    yy + rfy,
                                    xx + rfx].astype(data_type)
                               * kernel[koko,
                                        rc,
                                        rfy,
                                        rfx,
                                        kiki].astype(data_type),
                               axis=[rc, rfy, rfx],),
                           tag="conv2d_SOMA_tens")

    print("Preview of tensorized intrinsic:")
    print("================================")
    preview = te.create_schedule(conv_soma.op)
    print(tvm.lower(preview, [data, kernel], simple_mode=True))

    # Since no splitting is applied here, the strides are inferred automatically
    data_buffer = tvm.tir.decl_buffer(data.shape, data.dtype, name="data_b", offset_factor=1) # strides = [y*x,x,1])
    output_buffer = tvm.tir.decl_buffer(conv_soma.shape, conv_soma.dtype, name="conv_soma_b", offset_factor = 1)# , strides=[(y-fy+1)*(x-fx+1)*ki,(x-fx+1)*ki,ki,1])
    kernel_buffer = tvm.tir.decl_buffer(kernel.shape, kernel.dtype, name="kernel_b", offset_factor=1)# strides = [c*fy*fx*ki,fy*fx*ki,fx*ki,ki,1])

    def intrin_func(ins,outs):
        ib = tvm.tir.ir_builder.create()
        data_in, kernel_in = ins
        conv_out = outs
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "soma_wrapped_conv2d",
                x,      #uint32_t w
                y,      #uint32_t h
                c,      #uint32_t c
                fx,     #uint32_t fx
                fy,     #uint32_t fy
                ki*ko,  #uint32_t k
                y-fy+1, #uint32_t ox
                x-fx+1, #uint32_t oy
                1,      #uint32_t stride
                8,      #uint32_t precision
                0,      #uint32_t activation_function
                0,      #uint32_t zero_padding
                0,      #uint32_t shift_fixed_point
            )
        )
        return ib.get()
    return te.decl_tensor_intrin(conv_soma.op, intrin_func, binds={data: data_buffer,
                                                                   kernel: kernel_buffer,
                                                                   conv_soma: output_buffer})


intrinsic_size = 16

# Generate schedule to apply tensorization to:
data_orig = te.placeholder((B,C,Y,X),dtype=data_type, name="data_orig")
kernel_orig = te.placeholder((K,C,FY,FX), dtype=data_type, name="kernel_orig")
# Four dimensional tensor goes in, five dimensional tensor comes out
kernel_prepared = prepare_filter(kernel_orig, intrinsic_size)

in_batch, in_channel, in_height, in_width = data_orig.shape
num_filter_outer, channel, kernel_h, kernel_w, num_filter_inner = kernel_prepared.shape

rc = te.reduce_axis((0, in_channel), name="rfc")
rfy = te.reduce_axis((0, kernel_h), name="rfy")
rfx = te.reduce_axis((0, kernel_w), name="rfx")

conv = te.compute((in_batch,
                   num_filter_outer,
                   in_height - kernel_h + 1,
                   in_width - kernel_w + 1,
                   num_filter_inner,),
                  lambda b, ko, y, x, ki: te.sum( data_orig[in_batch,
                                                            rc,
                                                            y + rfy,
                                                            x + rfx].astype(data_type)
                                                  * kernel_prepared[ko,
                                                                    rc,
                                                                    rfy,
                                                                    rfx,
                                                                    ki].astype(data_type),
                                                  axis=[rc, rfy, rfx],),
                  tag="conv2d_SOMA_to_tensorize",
                  )

s = te.create_schedule(conv.op)

print("Generic Schedule for Element-wise Sum Compute to be tensorized:")
print("===============================================================")
tensorize_me = te.create_schedule(conv.op)
print(tvm.lower(tensorize_me, [data_orig, kernel_orig], simple_mode=True))
tensorize_me[conv].tensorize(conv.op.axis[1],intrin_conv2d("int8",intrinsic_size))
print("Schedule after tensorization")
print("============================")
print(tvm.lower(tensorize_me, [data_orig, kernel_orig], simple_mode=True))

lib = tvm.build(tensorize_me,[data_orig, kernel_orig],target_host="sirius")
file_name = "conv2d_soma.so"
lib.export_library(file_name,workspace_dir="/tmp/")