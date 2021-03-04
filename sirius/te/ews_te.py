from __future__ import absolute_import, print_function

import tvm
from tvm import te


def intrin_ews(ro,co,data_type):
    a = te.placeholder((ro,co), dtype=data_type, name="a")
    b = te.placeholder((ro,co), dtype=data_type, name="b")
    c = te.compute((ro,co), lambda i,j: a[i,j] + b[i,j], name="c")

    # Preview a generic schedule
    preview = te.create_schedule(c.op)
    print(tvm.lower(preview, [a, b, c], simple_mode=True))

    # Define buffers
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A")
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B")
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C")

    def intrin_func(ins, outs):
        # create IR builder
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",
                "ews",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
                ro,
                co,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


rows = 2
cols = 2
data_type = "int8"
# Create an instance
intrin_ews(rows,cols,data_type)
