# Simple file to generate an element wise sum ONNX file from pytorch
# sources:  *   https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
#           *   https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
#           *   https://tvm.apache.org/docs/tutorials/frontend/from_onnx.html

# Import pytorch dependencies
import torch
import torch.nn as nn
import torch.onnx
# Import tvm dependencies
import tvm
import tvm.relay as relay
import onnx
import numpy as np

from tvm.contrib import graph_runtime


class ElementWiseSum(nn.Module):
    # Generate a very simple module!
    def __init__(self):
        super(ElementWiseSum, self).__init__()

    def forward(self, x, y):
        return x.add(y)


# Generate two random input tensors
tensor_size = [2, 2]
data_type = torch.int8

tensor_a = torch.randint(0, 5, tensor_size, dtype=data_type)
tensor_b = torch.randint(0, 5, tensor_size, dtype=data_type)

# Create instance of model
model = ElementWiseSum()

# Run the neural network
result_sum = model(tensor_a,tensor_b)
print("A")
print(tensor_a)
print("B")
print(tensor_b)
print("sum")
print(result_sum)

# Generate onnx export parameters
input_names = ["A","B"]
output_names = ["sum"]

file_name = "ews.onnx"

# Export ONNX graph
print(f'...exporting ONNX file: {file_name}')
torch.onnx.export(model,(tensor_a,tensor_b), file_name, input_names=input_names, output_names=output_names)


# Import the graph into relay
print("...importing ONNX graph into Relay")
onnx_model = onnx.load(file_name)
x = np.array(tensor_a.numpy())
shape_dict = {"A": x.shape, "B": x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
target = tvm.target.Target("sirius")

"""
# THIS PART OF THE CODE FAILS
#   Check failed: t.lanes() == 1 (2 vs. 1) : do not yet support vector types
#   File "/home/josse/Thesis/tvm-fork/tvm-fork/src/target/source/codegen_c.cc", line 356
 
print("...compiling sirius module")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target)

runtime_module = graph_runtime.GraphModule(lib["default"](ctx))
runtime_module.set_input("A", tensor_a.numpy())
runtime_module.set_input("B", tensor_b.numpy())
runtime_module.run()
out = runtime_module.get_output(0, tvm.nd.empty(tensor_shape,dtype=data_type)).asnumpy()


file_name = "ews.so"
lib.export_library(file_name,workspace_dir="/tmp/")
"""