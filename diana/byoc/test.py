import relay_resnet20
from compare_outputs import get_gdb_output
from tvm.driver.tvmc.model import TVMCModel
from utils import (
                   tvmc_compile_and_unpack,
                   create_demo_file,
                   create_random_array,
                   adapt_gcc_opt,
                   make,
                   gdb
                  )
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
print(params)

model = TVMCModel(mod, params)
#init_value = -2
init_value = 1

# run on X86 to get demo_x86.txt
print("TEST: Running on X86")
device = "x86"
target = "c"
fusion = False
tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
create_demo_file(mod, init_value=init_value)
adapt_gcc_opt("Makefile.x86", 0)
make(device)
gdb(device, "build/demo", "gdb_demo_x86.sh")
print("TEST: parsing X86 output")

demo_x86 = get_gdb_output("demo_x86.txt")


# run on X86 to get demo_x86.txt
print("TEST: Running on Diana")
device = "pulp"
target = "soma_dory, c"
fusion = True
tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
create_demo_file(mod, init_value=init_value)
adapt_gcc_opt("Makefile.pulprt", 3)
make(device)
gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
print("TEST: parsing PULP output")

demo_pulp = get_gdb_output("demo.txt")

if np.ma.allequal(demo_x86,demo_pulp):
    print("TEST: PASS")
else:
    print("TEST: FAIL")
