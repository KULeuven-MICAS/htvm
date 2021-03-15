# TVM Tensor Expressions (TE) API examples

This folder contains some examples for using the TVM TensorExpressions (TE) API.
TE is a low level API for the TVM stack. 
It is available from C++ and Python. We currently only use the Python API.

## Contents

* Files: Examples for tensorization:
    * `ews_te.py` element-wise sum tensorized with a 2x2 matrix intrinsic using `tile`.
    * `ews_te_assign.py` variation of element-wise sum tensorization with a custom intrinsic.
    * `ews_te_soma.py` element-wise sum tensorized with a 4xHxC tensor intrinsic using `split`.
* In this readme:
    * An introduction to tensor expressions and a few useful links
    * A walkthrough of the `ews_te.py` example.
    * A quick note on strategies
---

## Introduction

TensorExpressions is based on [Halide](https://halide-lang.org/) and uses its decoupled compute/schedule paradigm.

TE is relatively well documented in [TVM's documentation](https://tvm.apache.org/docs/tutorials/index.html#tensor-expression-and-schedules). Some useful resources:
* [TE get started tutorial](https://tvm.apache.org/docs/tutorials/get_started/tensor_expr_get_started.html#sphx-glr-tutorials-get-started-tensor-expr-get-started-py)
* [List of all schedule primitives in TE](https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html#sphx-glr-tutorials-language-schedule-primitives-py)
* [TE Tensorization tutorial](https://tvm.apache.org/docs/tutorials/language/tensorize.html#sphx-glr-tutorials-language-tensorize-py)
* [Relay Op strategy tutorial](https://tvm.apache.org/docs/dev/relay_op_strategy.html): how to write your own Relay Strategy
* [Tuple input schedule tutorial](https://tvm.apache.org/docs/tutorials/language/tuple_inputs.html#sphx-glr-tutorials-language-tuple-inputs-py) (often used in Relay Strategies)


The Relay lowering process follows the relay strategy which defines what schedules for what operators to use in what case.
e.g. If you call an element-wise sum in relay, it should use the optimized element-wise sum operator for sirius
(the one that calls the element-wise sum in the HW library). In this context we use TE for:

* the definition of the Relay strategy.
  This basically tells TVM how to lower relay operator nodes to Target machine code. 
  Such a strategy is thus target dependent. 
  The Relay strategy for SIRIUS is described in `tvm/python/relay/op/strategy/sirius.py` and contains a lot of TE code.
  
* The optimized versions of operators. Often we use a tensorization primitive on this schedule.
  Tensorization replaces parts of the schedule with a certain intrinsic (e.g. a function call to the HW library).
  Optimized operators are thus also Target dependent. SIRIUS optimized operators are described in `tvm/python/topi/sirius/`
  
Besides the strategies and operators we created for SIRIUS it is instructive to look into operators for other targets defined in:
`tvm/python/topi/` and strategies for other targets in `tvm/python/relay/op/strategy`.

**Please note**:
**Working on operators in topi and the sirius strategy requires rerunning the python install command in the 
`tvm/python/setup.py` for the changes to reflect in the tvm package that python uses.**

---

## Tensorization walkthrough with `ews_te.py`

In this walkthrough we'll get a basic understanding of:
* How to define compute in TensorExpressions
* How to play with a schedule for this compute in TensorExpressions
  * How to use basic scheduling primitives like `split` and `tile` (other primitives are listed in [List of all schedule primitives in TE](https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html#sphx-glr-tutorials-language-schedule-primitives-py)).
  * How to tensorize a part of the schedule and map it to an intrinsic function.
  
Please also look at the full file in `sirius/te/ews_te.py`. The walkthrough will discuss following steps:
1. Defining the computation
2. Defining the schedule to be tensorized
3. Defining the computation and schedule for the intrinsic
4. Defining the right interface for the intrinsic 
5. Preparing the schedule to be tensorized
6. Performing tensorization on the prepared schedule

### Define the compute

First we need to specify what we want to compute. To represent a (possibly variable-shaped) tensor we use `te.placeholder`:
```python
A = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="A")
B = te.placeholder((ro,co,dim1,dim2), dtype=data_type, name="B")
```
Then we define the compute operation:
```python
C = topi.add(A,B)
```
We choose to use the topi implementation of add here, since it allows for quick testing of arbitrary sized tensors.
We could also define a 4 dimensional element-wise sum computation like this:
```python
C = te.compute((ro,co,dim1,dim2), lambda i,j,k,l: A[i,j,k,l] + B[i,j,k,l], name="T_add")
```

### Define the schedule to be tensorized

TVM will create a generic schedule that we can later alter with scheduling primitives.
Getting a generic schedule is really easy:
```python
s = te.create_schedule(C.op)
```
We can print a preliminary output of the schedule with this command:
```python
print(tvm.lower(s, [A, B, C], simple_mode=True))
```
Output:
```
primfn(A_1: handle, B_1: handle, T_add_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {T_add: Buffer(T_add_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             B: Buffer(B_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             A: Buffer(A_2: Pointer(int8), int8, [26, 26, 6, 2], [])}
  buffer_map = {A_1: A, B_1: B, T_add_1: T_add} {
  for (ax0: int32, 0, 26) {
    for (ax1: int32, 0, 26) {
      for (ax2: int32, 0, 6) {
        for (ax3: int32, 0, 2) {
          T_add_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)] = ((int8*)A_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)] + (int8*)B_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)])
        }
      }
    }
  }
}
```
We would like to offload a specific part of the computation to an intrinsic function. To do this, we first have to define the intrinsic:

### Define the computation and schedule for the intrinsic

We now define the computation of the intrinsic. Here we provide an intrinsic that performs element-wise sum on two matrices of size `(ro,co)`:
```python
a = te.placeholder((ro,co), dtype=data_type, name="a")
b = te.placeholder((ro,co), dtype=data_type, name="b")
c = te.compute((ro,co), lambda i,j: a[i,j] + b[i,j], name="c")
preview = te.create_schedule(c.op)
```
If we preview the schedule we get this (for `ro, co = (2, 2)`):
```
primfn(a_1: handle, b_1: handle, c_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {c: Buffer(c_2: Pointer(int8), int8, [2, 2], []),
             b: Buffer(b_2: Pointer(int8), int8, [2, 2], []),
             a: Buffer(a_2: Pointer(int8), int8, [2, 2], [])}
  buffer_map = {a_1: a, b_1: b, c_1: c} {
  for (i: int32, 0, 2) {
    for (j: int32, 0, 2) {
      c_2[((i*2) + j)] = ((int8*)a_2[((i*2) + j)] + (int8*)b_2[((i*2) + j)])
    }
  }
}
```
Next we need to define buffers that will match the schedule to be tensorized.
```python
Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[stride,1])
Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[stride,1])
Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[stride,1])
```
They should have the same shape and data types as the tensors in the compute of the intrinsic.
`offset_factor` is used by TVM to allow for implementing vectorized loading. It is best left at 1 for our purposes.
`strides` is an important field that needs to match the strides in the schedule to be tensorized. 
You can see an example of bad strides here:  https://discuss.tvm.apache.org/t/te-tensorize-elementwise-sum/9335/3.
We will discuss how to derive the correct value of the strides in the next section.

### Defining the right interface for the intrinsic

We have to tell TVM how to compile our intrinsic to C code, as:

* we want TVM to use the right C function
* TVM passes tensors to the C function
* some values for the C functions need to be inserted in the C code call.

We can tell TVM to call our function by making it emit a `tvm.tir.call_extern` function in the schedule.
This directly defines what C code will be created by the backend.

```python
    def intrin_func(ins, outs):
        # create IR builder
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "int32",                    # output data type (= C function return value)
                "soma_wrapped_ews",         # intrinsic function name (= C function call name)
                aa.access_ptr("r"),         # arguments (= C function arguments):
                bb.access_ptr("r"),         #   will create a pointer to a tensor buffer
                cc.access_ptr("w"), 
                a.shape[0], # width,here 2  #   Some values that need to be inserted in the C code call
                a.shape[1], # height, here 2
                1,          # channels      #   the channel value here is hardcoded. You can check 
                8,          # precision     #   ews_te_soma.py for an example with variable sized channels.
            )
        )
        return ib.get()
```
The above code would produce the following C code if it were built:
```C
(void) soma_wrapped_ews((int_8t *) placeholder + (0)), (int_8t *) placeholder1 + (0)), (int_8t *) T_add + (0)), 2, 2, 1, 8)
```
**Note**: in the current `sirius` backend the return value (here `int32`) is not used to produce C code.

In a last step, we need to map the buffers to the defined intrinsic function by returning:
```python
te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})
```

### Preparing the schedule to be tensorized

We would like to insert our intrinsic to speed up the 2 innermost loops of the original schedule:
```
for (ax2: int32, 0, 6) {
  for (ax3: int32, 0, 2) {
    T_add_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)] = ((int8*)A_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)] + (int8*)B_2[((((ax0*312) + (ax1*12)) + (ax2*2)) + ax3)])
  }
}
```
However, we can see that the dimensions of the two innermost (for loop goes to 6 and 2) 
loops don't match the dimensions of the intrinsic (for loop goes to 2 and 2)
Therefore we need to split up the for loops in multiple parts.

Here we use the `tile` primitive to perform the splitting of two for loops (=two axes) at once. 
```python
xo, yo, xi, yi = s[C].tile(C.op.axis[-2], C.op.axis[-1],x_factor=2,y_factor=2)
```
We use negative indexing to select the two innermost for loops (the two innermost axes):`C.op.axis[-2]` and `C.op.axis[-1]`
It is very important here that the x_factor and y_factor here match the dimensions of the instrinsic so we can tensorize later.
`xo, yo, xi, yi` match handles for the respective inner and outer loops that are the result of tiling.
After tiling we get this schedule:
```
primfn(A_1: handle, B_1: handle, T_add_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {T_add: Buffer(T_add_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             B: Buffer(B_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             A: Buffer(A_2: Pointer(int8), int8, [26, 26, 6, 2], [])}
  buffer_map = {A_1: A, B_1: B, T_add_1: T_add} {
  for (ax0: int32, 0, 26) {
    for (ax1: int32, 0, 26) {
      for (ax2.outer: int32, 0, 3) {
        for (ax2.inner: int32, 0, 2) {
          for (ax3.inner: int32, 0, 2) {
            T_add_2[(((((ax0*312) + (ax1*12)) + (ax2.outer*4)) + (ax2.inner*2)) + ax3.inner)] = ((int8*)A_2[(((((ax0*312) + (ax1*12)) + (ax2.outer*4)) + (ax2.inner*2)) + ax3.inner)] + (int8*)B_2[(((((ax0*312) + (ax1*12)) + (ax2.outer*4)) + (ax2.inner*2)) + ax3.inner)])
          }
        }
      }
    }
  }
}
```
You can see that ax3 only has an inner loop. This is because the dimension of 2 already matched the factor of 2, so TVM
did not perform any splitting on this axis.

From this schedule we can also see what the strides are for the respective axes.
In this case you can see a `(ax2.inner*2)` which means that we need a stride of `2` there.
We can also see what the value has to be of the second stride: `ax3.inner`, there's no factor, so you can leave the stride `1` here.

To get the strides in a variable fashion, you can derive them from the extents of the axis (the iterations in the for loop):
```python
stride = s[C].op.axis[-1].dom.extent
```
### Performing tensorization

Now that we have prepared the general schedule by splitting up the for loops and by defining the intrinsic, we can start tensorizing.
```python
s[C].tensorize(xi, intrin_ews(rows,cols,data_type,stride=stride))
```
`xi` is the outermost axis we want to tensorize over. 
In the above line of code we also immediately create an intrinisc with the correct stride.

This outputs the following schedule:
```
primfn(A_1: handle, B_1: handle, T_add_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {T_add: Buffer(T_add_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             B: Buffer(B_2: Pointer(int8), int8, [26, 26, 6, 2], []),
             A: Buffer(A_2: Pointer(int8), int8, [26, 26, 6, 2], [])}
  buffer_map = {A_1: A, B_1: B, T_add_1: T_add} {
  for (ax0: int32, 0, 26) {
    for (ax1: int32, 0, 26) {
      for (ax2.outer: int32, 0, 3) {
        @tir.call_extern("soma_wrapped_ews", @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), A_2, (((ax0*312) + (ax1*12)) + (ax2.outer*4)), 4, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), B_2, (((ax0*312) + (ax1*12)) + (ax2.outer*4)), 4, 1, dtype=handle), @tir.tvm_access_ptr(@tir.type_annotation(, dtype=int8), T_add_2, (((ax0*312) + (ax1*12)) + (ax2.outer*4)), 4, 2, dtype=handle), 2, 2, 1, 8, dtype=int32)
      }
    }
  }
}
```
Tensorization is now finished!

## From here

The above walkthrough was based on ews_te.py. A more advanced tensorization approach can be found in `ews_te_soma.py`

If you feel comfortable with these TE examples you can take a look at Relay strategies, where you use TE to define what
optimized operation to compile to based on several conditions.

---

# Relay op strategy

For a complete introduction to Relay operator strategies we refer to [Relay Op strategy tutorial](https://tvm.apache.org/docs/dev/relay_op_strategy.html).

Our current efforts for creating a Relay Operator strategy can be found in: `tvm/python/relay/op/strategy/sirius.py`
It refers to code that is written in `tvm/python/topi/sirius/`.
For an element-wise sum, you should look into `injective.py` as an element-wise operation is an injective operation 
(injective operations are defined by TVM to have the same tensor output shape as the input(s)).
