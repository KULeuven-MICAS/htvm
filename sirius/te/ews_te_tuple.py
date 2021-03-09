# Calculate Simple 4x4 element wise sum and tensorize over  2x2 matrices
# Also look into:
#
# *  https://tvm.apache.org/docs/tutorials/language/tensorize.html
# *  https://discuss.tvm.apache.org/t/te-tensorize-elementwise-sum/9335/3

from __future__ import absolute_import, print_function

import tvm
from tvm import te
from tir import 


def intrin_ews(ro,co,data_type):
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
    # TODO make strides variable!
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[4,1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[4,1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[4,1])

    def intrin_func(ins, outs):
        # Building TIR code to load ins[1] in outs.
        #
        # Both input_buffer and output_buffer are of type tvm.tir.Buffer
        _, input_buffer = ins
        output_buffer = outs[0]
        # We build an expression that reads from the input buffer.
        input_read_expr = input_buffer.vload([0, 0])
        # We build a statement that writes the loaded values into the output buffer.
        output_write_statement = output_buffer.vstore([0, 0], input_read_expr)
        # We return the expected statement.
        return output_write_statement

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


rows = 2
cols = 2
data_type = "float32"
# Create an instance
intrinsic = intrin_ews(rows,cols,data_type)

ro = 4
co = 4
# Create a tensorizable schedule
A = te.placeholder((ro,co), dtype=data_type, name="A")
B = te.placeholder((ro,co), dtype=data_type, name="B")
C = te.compute((ro,co), lambda i,j: A[i,j] + B[i,j], name="C")
# Create a vanilla schedule
s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1],x_factor=2,y_factor=2)
print("Larger schedule to apply tensorization of the Generic Schedule (after tiling):")
print("==============================================================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))
# Tensorize!
s[C].tensorize(xi, intrinsic)
print("After tiling and applying the tensorization:")
print("============================================")
print(tvm.lower(s, [A, B, C], simple_mode=True))
