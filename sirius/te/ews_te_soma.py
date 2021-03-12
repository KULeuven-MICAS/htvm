# This version of element wise sum also supports a channel dimension and will only split one loop to be fixed.
# This is the way of working for SOMA, as the hardware library will unroll the channel and height dimensions of
# the element wise sum temporally already
#
# Also look into:
# *  https://tvm.apache.org/docs/tutorials/language/tensorize.html
# *  https://discuss.tvm.apache.org/t/te-tensorize-elementwise-sum/9335/3

import tvm
from tvm import te
from tvm import topi

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
    # Offset factor --> so TVM can optimize for vectorized buffering
    # Stride        --> see TVM discuss post above
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


# Dimensions of tensorization intrinsic
width = 2
data_type = "int8"

# Dimensions of tensor to be tensorized
ro = 26
co = 20
dim1 = 14
dim2 = 16

# Create a tensorizable schedule
A = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="A")
B = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="B")
# C = te.compute((ro,co,dim1,dim2), lambda i,j,k,l: A[i,j,k,l] + B[i,j,k,l], name="T_add")
# Using topi.add implementation here to reflect schedule of relay graph instead of custom operation
C = topi.add(A, B)

# Create a vanilla schedule
s = te.create_schedule(C.op)
print("Larger schedule to apply tensorization of the Generic Schedule (before split):")
print("==============================================================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))

# indexing axes negatively to split over third innermost axis
yo, yi = s[C].split(C.op.axis[-3], factor=2)
print("Larger schedule to apply tensorization of the Generic Schedule (after split):")
print("==============================================================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))

# Get extent of most innermost original axis to act as a stride parameter in the tensorized version
stride_innermost = s[C].op.axis[-1].dom.extent
stride_outermost = s[C].op.axis[-2].dom.extent * s[C].op.axis[-1].dom.extent

# Tensorize!
s[C].tensorize(yi, intrin_ews_soma(width, data_type,stride_outermost=stride_outermost, stride_innermost=stride_innermost))
print("After splitting and applying the tensorization:")
print("============================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))