# This is a simple example of a CONV2D operation for SOMA


import tvm
from tvm import te
from tvm import topi


# Tensor Dimensions

B = 2       # Input batch (also called N sometimes)
C = 3       # Input channels
X = 224     # Input width (also called W sometimes)
Y = 224     # Input height (also called H sometimes)

FX = 5      # Filter width
FY = 5      # Filter height
K = 32      # Filter output channels

data_type = "int8"

# Demonstrating the standard topi implementation of Conv2D
# Input data layout is NCHW or BCYX
data = te.placeholder((B, C, Y, X), dtype=data_type, name="data")
# Input filter layout is KCFyFx or NCFyFx
kernel = te.placeholder((K, C, FY, FX), dtype=data_type, name="kernel")
conv = topi.nn.conv2d(data, kernel, 1, 0, 1, layout="NCHW", out_dtype=data_type)

s = te.create_schedule(conv.op)
print(tvm.lower(s, [data, kernel], simple_mode=True))


# Transform the input data so it has the right dimensions:
Ki = 16
# Calculate the array size and modify length accordingly
d_shp = kernel.shape
K = int(d_shp[0])
assert (K % Ki == 0), f"K size must be a multiple of {Ki}"
# "split" Kernel dimension K in two pieces Ko,Ki with reshape:
Ko = int(K / Ki)
reshaped = topi.reshape(kernel, (Ko, Ki, d_shp[1], d_shp[2], d_shp[3]))
# New Kernel layout is Ko,Ki,C,Fx,Fy --> transpose to Ko,C,Fx,Fy,Ki
kernel_prepared = topi.transpose(reshaped, (0, 2, 3, 4, 1))


# Create a vanilla schedule
s = te.create_schedule(conv.op)
print("Example of the generic Conv2D schedule")
print("======================================")
print(tvm.lower(s, [data, kernel], simple_mode=True))

data_prepared = te.placeholder((C, Y, X), dtype=data_type, name="data_prepared")

in_channel, in_height, in_width = data_prepared.shape
num_filter_outer, channel, kernel_h, kernel_w, num_filter_inner = kernel_prepared.shape

# Define reduce axes for summation operation
rc = te.reduce_axis((0, in_channel), name="rfc")
rfy = te.reduce_axis((0, kernel_h), name="rfy")
rfx = te.reduce_axis((0, kernel_w), name="rfx")

# In this example padding is not implemented
comp = te.compute((Ko, Y-FY+1, X-FX+1, Ki),
                  lambda ko, y, x, ki: te.sum(
                      data_prepared[rc,
                                    y + rfy,
                                    x + rfx].astype(data_type)
                      * kernel_prepared[ko,
                                        rc,
                                        rfy,
                                        rfx,
                                        ki].astype(data_type),
                      axis=[rc, rfy, rfx],),
                  tag="conv2d_nchw",
                  )

s = te.create_schedule(comp.op)

# Merge reshape and transpose steps to avoid unnecessary copy of reshape
s[reshaped].compute_inline()

print("Custom SOMA-compatible Conv2D schedule")
print("======================================")
print(tvm.lower(s, [data_prepared, kernel], simple_mode=True))
