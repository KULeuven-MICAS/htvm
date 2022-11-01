import relay_resnet20
from tvm.driver.tvmc.model import TVMCModel
import utils
import numpy as np

import tvm.relay as relay
from relay_simple import create_model
#from relay_resnet20 import create_model
#from mlperf_tiny.relay_dae import create_model
#from mlperf_tiny.relay_ds_cnn import create_model
#from mlperf_tiny.relay_mobilenet import create_model
#from mlperf_tiny.relay_resnet import create_model

from utils import relay_soma_conv2d
import tvm


#import resnet20 model

# for reproducability
np.random.seed(0)
mod, params = create_model(8)
#mod, params = create_model(2)

model = TVMCModel(mod, params)
#init_value = -2
init_value = 1


# run on X86 to get demo_x86.txt
print("TEST: Compiling for X86")
device = "x86"
target = "c"
fusion = False
utils.tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
utils.create_demo_file(mod, init_value=init_value)
utils.adapt_gcc_opt("Makefile.x86", 0)
utils.make(device)
print("TEST: obtaining X86 output")
result_x86 = utils.gdb(device, "build/demo", "gdb_demo_x86.sh")
print(result_x86)

# run on X86 to get demo_x86.txt
print("TEST: compiling for Diana")
device = "pulp"
target = "soma_dory, c"
fusion = True
utils.tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
utils.create_demo_file(mod, init_value=init_value)
utils.adapt_gcc_opt("Makefile.pulprt", 3)
utils.make(device)
result_pulp = utils.gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
print("TEST: obtaining Diana output")
print(result_pulp)

print("Final Results")
print("=============")
print("X86 output:")
print(result_x86)
print("Diana output:")
print(result_pulp)
if np.ma.allequal(result_x86,result_pulp):
    print("TEST: PASS")
else:
    print("TEST: FAIL")

