import tvm
from tvm import te

# Define vector inputs a and b
in_shape = (6,)
dtype = "int32"

a = te.placeholder(in_shape, dtype, name="vec_a")
b = te.placeholder(in_shape, dtype, name="vec_b")

# Define computation and output vector c
out_shape = in_shape
c = te.compute(out_shape, lambda i: a[i] + b[i], name='vec_add')

# Create a schedule for the computation
schedule = te.create_schedule(c.op)
arguments = [a, b]
print(tvm.lower(schedule, arguments, simple_mode=True))

# Define intrinsic
def intrin_vec_add(len, dtype="int32"):
    int_in_shape = (len,)
    int_out_shape = int_in_shape
    d = te.placeholder(int_in_shape, dtype=dtype, name="d")
    e = te.placeholder(int_in_shape, dtype=dtype, name="e")
    f = te.compute(int_out_shape, lambda i: d[i] + e[i], name="f")

    sched = te.create_schedule(f.op)
    argum = [d, e]
    print(tvm.lower(sched, argum, simple_mode=True))

    db = tvm.tir.decl_buffer(d.shape, d.dtype, name="d_buf", offset_factor=1, strides=[1])
    eb = tvm.tir.decl_buffer(e.shape, e.dtype, name="e_buf", offset_factor=1, strides=[1])
    fb = tvm.tir.decl_buffer(f.shape, f.dtype, name="f_buf", offset_factor=1, strides=[1])

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        dd, ee = ins
        ff = outs[0]
        ib.emit(tvm.tir.call_extern(
            "int32",
            "double_add_intrinsic",
            dd.access_ptr("r"),
            ee.access_ptr("r"),
            ff.access_ptr("w")
        ))
        return ib.get()

    return te.decl_tensor_intrin(f.op, intrin_func, binds={d: db, e: eb, f: fb})

intrin_length = 2
outer, inner = schedule[c].split(c.op.axis[0], factor=intrin_length)
print(tvm.lower(schedule, arguments, simple_mode=True))
schedule[c].tensorize(inner, intrin_vec_add(intrin_length))
print(tvm.lower(schedule, arguments, simple_mode=True))
