# TVM Relay API examples

This folder contains some examples for using the TVM Relay API for constructing or importing neural network computation graphs.

## Contents
* Directly constructing a simple execution graph in Relay.
  * `ews_mat.py` element-wise matrix addition for use with TVM LLVM-backend run on your own machine
  * `ews_mat_micro.py`, element-wise matrix addition for use with microTVM C backend for external compilation. E.g. for deployment on Sirius.

---

## How it works

In this section we will go over the execution graph of a simple element-wise matrix addition of two matrices, directly constructed in Relay's python API.
It was largely based on [this quickstart tutorial](https://tvm.apache.org/docs/tutorials/get_started/relay_quick_start.html), and [this TVMconf2020 microTVM tutorial](https://github.com/areusch/microtvm-blogpost-eval/blob/master/tutorial/standalone_utvm.ipynb) (showcased in [this video](https://www.youtube.com/watch?v=pp5Xwhlu9Bk)).

We discuss two working methods:
1. Using TVM's LLVM backend and minimal runtime. In this way we can run the constructed graph directly on our workstation computer (`ews_mat.py`).
2. Using TVM's microTVM C emitter backend for use on microcontroller platforms. In this way we can use an external C compiler that supports the target microcontroller (`ews_mat_micro.py`).


First we need to import some libraries in python.
```python
import tvm
import tvm.relay as relay
import numpy as np

from tvm.contrib import graph_runtime # Only for execution on a workstation CPU.
```

### Construct the execution graph

While Relay is often used as an IR for importing complex neural networks,
it can also be used to construct execution graphs directly.

First we define placeholder dimensions, types and then later the actual placeholders that will contain our input data. The placeholders are of the `tvm.relay.Var` type.
```python
tensor_shape = (2,2)
data_type = "int8"

a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))
```

TVM stores its intermediate representations in a `tvm.ir.IRModule` container.
While such a container for use with Relay typically contains a `tvm.relay.Function` object, we can also use the `from_expr()` method on an `IRModule` object to create an `IRModule` from a simpler `tvm.relay.Expr`.
The `tvm.relay.Expr` we'll use in this example is a simple addition with `tvm.relay.add()`.
```python
sum_expr = relay.add(a,b)
module = tvm.ir.IRModule()
module = module.from_expr(sum_expr)
```

### Define the deployment target

In this part we define a different deployment target for the two aforementioned deployment methods. In TVM it is quite easy to switch between deployment targets, making it very easy to construct **golden models**.
You can e.g. run an LLVM-compiled golden model of a certain neural network on your workstation, and compare the outputs of this model with a version that is compiled for a microcontroller

```python
# For deployment on a workstation:
target = tvm.target.Target("llvm")
# For deployment on a microcontroller
target = tvm.target.Target("c -march=rv32imf -link-params -runtime=c -system-lib=1 ")
```
For the microcontroller the c emitter backend is used which has some options.
* `-march=rv32imf`: Generate C code for a RISC-V rv32imf ISA.
* `-runtime=c` : build code for the TVM C runtime (i.e. the bare-metal compatible one)
* `link-params`: link supplied model parameters (e.g. weights) as constants in the generated code
* `system-lib` : Build a "system library." In deployments, the system library is pre-loaded into the runtime, rather than a library that needs to be loaded e.g. from a file. This is the simplest configuration for a bare-metal microcontroller, so we use it here.

### Build the target

Now that the execution graph is defined and we have chosen a target, we can build the C code or the LLVM IR representation.
```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(module, target)
```
The outputted json_execution graph, compiled functions and simplified parameters can be extracted from this step, but you'll get a depreciation warning if you try to do that.
```python
# This method is depreciated
with tvm.transform.PassContext(opt_level=3):
  graph_json, compiled_model, simplified_params = relay.build(module, target=target)
```
*microTVM only:* It should be noted that the [microTVM tutorial](https://github.com/areusch/microtvm-blogpost-eval/blob/master/tutorial/standalone_utvm.ipynb) also disables vectorization in TIR:
```python
with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize':True}):
    lib = relay.build(module, target=target)
```
*LLVM-only:* From the `lib` variable you can also directly compile a shared library (`.so` file on linux)

```python
lib.export_library("compiled_lib.so")
```

### *LLVM-only* - Run the built model

To run the built model we first define two random input tensors:
```python
a = np.random.randint(0,5,size=tensor_shape).astype(data_type)
b = np.random.randint(0,5,size=tensor_shape).astype(data_type)
```
Now we can define a runtime environment for the built model to run on.
This needs a context definition to make sure it's running on the cpu:
```python
ctx = tvm.cpu()
```
Then we can construct an appropriate graph runtime and connect the random input tensors we created:
```python
runtime_module = graph_runtime.GraphModule(lib["default"](ctx))
runtime_module.set_input("a", a)
runtime_module.set_input("b", b)
```
The output can then be copied over to an empty numpy array:
```python
out = runtime_module.get_output(0, tvm.nd.empty(tensor_shape,dtype=data_type)).asnumpy()
```
You can even check whether the compiled instructions to TVM make sense:
```python
print("TVM and Numpy match?")
print((out == a+b).all())
```
### *microTVM only* - Output and inspect emitted C code

To output various c-files for microTVM compilation it is possible get the source in multiple ways.
It should be noted that in this case lib.export_library will most likely throw an error, because it cannot find certain header files.(related to DLPack, and if you solve this error, it will ask you to include the tvm c runtime header files.)
However, even though compilation with `g++` will fail, you can still see all outputted c files.
```python
# This only outputs the generated C-files for the modules.
lib.get_source()
# All necessary c-files are copied to the workspace but some headers might be missing.
# This results in runtime errors (failing compilation) and will halt your script.
lib.export_library(file_name="shared_library.so",workspace_dir="workspace_directory")
```
### *microTVM only* - Compile the C code and flash to device

After creating C code for the microcontroller, you want to compile the C
