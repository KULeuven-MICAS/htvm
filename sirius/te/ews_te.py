# Calculate Simple 4x4 element wise sum and tensorize over  2x2 matrices
# Also look into:
#
# *  https://tvm.apache.org/docs/tutorials/language/tensorize.html
# *  https://discuss.tvm.apache.org/t/te-tensorize-elementwise-sum/9335/3

from __future__ import absolute_import, print_function

import tvm
from tvm import te


def intrin_ews(ro,co,data_type,stride):
    a = te.placeholder((ro,co), dtype=data_type, name="a")
    b = te.placeholder((ro,co), dtype=data_type, name="b")
    c = te.compute((ro,co), lambda i,j: a[i,j] + b[i,j], name="c")

    # Preview a generic schedule
    print("Generic Schedule for Element-wise Sum Compute to be tensorized:")
    print("===============================================================")
    preview = te.create_schedule(c.op)
    print(tvm.lower(preview, [a, b, c], simple_mode=True))

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

# Dimensions of tensorization
rows = 2
cols = 2
data_type = "float32"
# Create an instance


# Dimensions of tensor to be tensorized
ro = 26
co = 26
dim1 = 6
dim2 = 2
# Create a tensorizable schedule
A = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="A")
B = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="B")
C = te.compute((ro,co,dim1,dim2), lambda i,j,k,l: A[i,j,k,l] + B[i,j,k,l], name="C")
# Create a vanilla schedule
s = te.create_schedule(C.op)
print("Larger schedule to apply tensorization of the Generic Schedule (before tiling):")
print("==============================================================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))
# indexing axes negatively to tile over two innermost axes
xo, yo, xi, yi = s[C].tile(C.op.axis[-2], C.op.axis[-1],x_factor=2,y_factor=2)
print("Larger schedule to apply tensorization of the Generic Schedule (after tiling):")
print("==============================================================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))
# Tensorize!
print(s[C].op.axis[-1])
# Get extent of most innermost original axis to act as a stride parameter in the tensorized version
stride = s[C].op.axis[-1].dom.extent
s[C].tensorize(xi, intrin_ews(rows,cols,data_type,stride=stride))
print("After tiling and applying the tensorization:")
print("============================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))
