#*************************************************************************
# (uTVM) In this Relay example I try to implement a simple matrix addition
#*************************************************************************
# References: https://tvm.apache.org/docs/tutorials/get_started/relay_quick_start.html
#             https://github.com/areusch/microtvm-blogpost-eval/blob/master/tutorial/standalone_utvm.ipynb

import tvm
import tvm.relay as relay
import numpy as np

# create tensor variables with dimensions (batch,channels,x,y)
# tensor_shape = (1,1,8,8)

# Or create a 2D-matrix
tensor_shape = (20,20,8)
data_type = "int8"

# Construct the variables --> tvm.relay.Var type
a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))

# Then we tell it to add the two variables --> tvm.relay.Expr type
sum_expr = relay.add(a,b)

# Now create an IRModule from the tvm.relay.Expr file
module = tvm.ir.IRModule()
module = module.from_expr(sum_expr)

print(module
      )

# Define a target for compilation and a runtime context for the embedded device

# c             --> emit C code
# march=rv32imf --> Generate C code for a RISC-V rv32imf ISA.
# runtime=c     --> build code for the TVM C runtime (i.e. the bare-metal compatible one)
# link-params   --> link supplied model parameters as constants in the generated code
# system-lib    --> Build a "system library." In deployments, the system library is pre-loaded into the runtime, rather than a library that needs to be loaded e.g. from a file. This is the simplest configuration for a bare-metal microcontroller, so we use it here.
# target = tvm.target.Target("c -march=rv32imf -link-params -runtime=c -system-lib=1 ")

# Optimize (?)  and build the relay code:
with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize':True}):
    #graph_json, compiled_model, simplified_params = relay.build(module, target=target)
    lib = relay.build(module, target="c")

# All necessary c-files are copied to the workspace but some headers might be missing.
# This results in runtime errors (failing compilation) and will halt your script.

file_name = "ews.so"
lib.export_library(file_name,workspace_dir="/tmp/tvm_workspace")
