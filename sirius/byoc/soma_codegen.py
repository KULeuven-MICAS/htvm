#*************************************************************************
# (LLVM) In this Relay example I try to implement a simple matrix addition
#*************************************************************************

import tvm
import tvm.relay as relay
import tvm.relay.op.contrib.soma as soma
import numpy as np

from tvm.contrib import graph_runtime
from aot_tvm import tvm_compile
import os

# create tensor variables with dimensions (batch,channels,x,y)
# tensor_shape = (1,1,8,8)
# Or create a 2D-matrix
tensor_shape = (2,2)
data_type = "int8"

# Construct the variables --> tvm.relay.Var type
a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))

# Then we tell it to add the two variables --> tvm.relay.Expr type
sum_expr = relay.add(a, b)

# Now create an IRModule from the tvm.relay.Expr file
module = tvm.ir.IRModule()
module = module.from_expr(sum_expr)

# As in documentation: https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm#bring-dnnl-to-tvm-annotation-rules

#module = relay.transform.MergeComposite(soma.pattern_table)(module)
module = relay.transform.AnnotateTarget(["soma"])(module)
module = relay.transform.MergeCompilerRegions()(module)
module = relay.transform.PartitionGraph()(module)

# Define a target for compilation
target = tvm.target.Target("c")

# Optimize (?)  and build the relay code:
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(module, target)

# At this point you can export the library as a shared library file (.so)
#lib.export_library("compiled_lib.so")

## Put the generated library to work in a runtime module with cpu context
#ctx = tvm.cpu()
#runtime_module = graph_runtime.GraphModule(lib["default"](ctx))
#
## create two random tensors...
#a = np.random.randint(0,5,size=tensor_shape).astype(data_type)
#b = np.random.randint(0,5,size=tensor_shape).astype(data_type)
#
## ... and set them as input for the module
#runtime_module.set_input("a", a)
#runtime_module.set_input("b", b)
#
## Run the runtime module
#runtime_module.run()
#
## Get the outputs and copy them to an empty numpy array
#out = runtime_module.get_output(0, tvm.nd.empty(tensor_shape,dtype=data_type)).asnumpy()
#
## Print the in and outputs and compare to see if TVM made something that makes sense
#print("a:")
#print(a)
#print("b:")
#print(b)
#print("Numpy:")
#print(a+b)
#print("TVM:")
#print(out)
#print("TVM and Numpy match?")
#print((out == a+b).all())


build_directory = "/tmp/"

library = lib.get_lib()
json = lib.get_json()
new_params = lib.get_params()

try:
    library.export_library(
        "soma.so", workspace_dir=build_directory
    )
except Exception:
    pass

tvm_compile(
    json,
    new_params,
    os.path.join(build_directory, "soma_main.c"),
    os.path.join(build_directory, "soma.h"),
    verbose=True,
    get_per_node=False,
    add_perf_counter=True,
    perf_type="c",
)

