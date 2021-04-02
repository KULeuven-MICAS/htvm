# Relay high-level API compilation of Conv2D

import tvm
import tvm.relay as relay

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
conv = relay.nn.conv2d(activations, weights,kernel_size=(fx,fy), padding=(2,2,2,2))

func = relay.Function(relay.analysis.free_vars(conv),conv)

mod = tvm.ir.IRModule().from_expr(func)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target="sirius", target_host="sirius")

file_name = "conv2d.so"
lib.export_library(file_name, workspace_dir="/tmp")