#******************************************************************
# In this Relay example I try to implement a simple matrix addition
#******************************************************************

import tvm
import tvm.relay as relay
import numpy as np

from tvm.contrib import graph_runtime

# create tensor variables with dimensions (batch,channels,x,y)
#tensor_shape = (1,1,8,8)
tensor_shape = (2,2)
data_type = "int8"

a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))

# This creates a tvm.relay.Expr type
sum_expr = relay.add(a,b)

# Now create an IRModule
mod = tvm.ir.IRModule()
mod = mod.from_expr(sum_expr)

# Define a target for compilation and a runtime context for cpu
target = tvm.target.Target("llvm")
ctx = tvm.cpu()

# Build the relay code:
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target)

# Put the generated library to work in a module
module = graph_runtime.GraphModule(lib["default"](ctx))

# create two random tensors...
a = np.random.randint(0,5,size=tensor_shape).astype(data_type)
b = np.random.randint(0,5,size=tensor_shape).astype(data_type)

# ... and set them as input for the module 
module.set_input("a", a)
module.set_input("b", b)

module.run()

out = module.get_output(0, tvm.nd.empty(tensor_shape,dtype=data_type)).asnumpy()

print("a:")
print(a)
print("b:")
print(b)
print("Numpy:")
print(a+b)
print("TVM:")
print(out)
