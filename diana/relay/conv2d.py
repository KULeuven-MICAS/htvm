# Relay high-level API compilation of Conv2D

import tvm
import tvm.relay as relay
import numpy as np

b = 2       # Input batch (also called N sometimes)
c = 3       # Input channels
x = 256     # Input width (also called W sometimes)
y = 256     # Input height (also called H sometimes)

fx = 5      # Filter width
fy = 5      # Filter height
k = 32      # Filter output channels

data_type = "int8"
act_shape = (b, c, y, x)
wgt_shape = (k, c, fx, fy)

activations = relay.var("act", tvm.relay.TensorType(act_shape, data_type))
weights = relay.var("wgt", tvm.relay.TensorType(wgt_shape, data_type))

# Only provide spatial dimension to kernel_size, otherwise, it doesn't work!
# Make padding "same" --> padding should be equal to fx // 2
conv = relay.nn.conv2d(activations, weights,kernel_size=(fx,fy), padding=fx // 2)

func = relay.Function(relay.analysis.free_vars(conv),conv)

print(func)

mod = tvm.ir.IRModule().from_expr(func)

random_weights = np.random.randint(0,5,size=wgt_shape).astype(data_type)
random_data = np.random.randint(0,5,size=act_shape).astype(data_type)

weights_params = tvm.nd.array(random_weights)
activations_params = tvm.nd.array(random_data)

params = {"act": random_data,
          "wgt": random_weights}


with tvm.transform.PassContext(opt_level=3):
    graph, lib, weights = relay.build(mod, target="sirius", params=None)

print(graph)
print(lib)
print(weights)

with open('/tmp/graph.json','w') as file:
    file.write(graph)
with open('/tmp/weights.json','w') as file:
    file.write(str(weights))

file_name = "conv2d.so"
lib.export_library(file_name, workspace_dir="/tmp")