# Simple file to generate an element wise sum ONNX file from pytorch
# sources:  *   https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html
#           *   https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


import torch
import torch.nn as nn
import torch.onnx


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

# Export ONNX graph
torch.onnx.export(model,(tensor_a,tensor_b),"ews.onnx",input_names=input_names, output_names=output_names)

