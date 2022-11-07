import relay_resnet20
from tvm.driver.tvmc.model import TVMCModel
import utils
import numpy as np
import profiler

import tvm.relay as relay
#from relay_simple import create_model
#from relay_resnet20 import create_model
from single_layer.relay_dense import create_model
#from single_layer.relay_conv2d import create_model
#from single_layer.relay_dw_conv2d import create_model
#from mlperf_tiny.relay_dae import create_model
#from mlperf_tiny.relay_ds_cnn import create_model
#from mlperf_tiny.relay_mobilenet import create_model
#from mlperf_tiny.relay_resnet import create_model


# for reproducability
np.random.seed(0)

# Set precision to 2 for triggering analog core
precision = 8

mod, params = create_model(precision)

model = TVMCModel(mod, params)
#init_value = -2
init_value = 1

# int2 verification is not available on X86
if precision != 2:
    # run on X86
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

# run on Diana
print("TEST: compiling for Diana")
device = "pulp"
target = "soma_dory, c"
fusion = True
utils.tvmc_compile_and_unpack(model, target=target, fuse_layers=fusion)
# Add analog boot code in case of precision
if precision == 2:
    utils.create_demo_file(mod, init_value=init_value, boot_analog=True)
else:
    utils.create_demo_file(mod, init_value=init_value)
utils.adapt_gcc_opt("Makefile.pulprt", 3)
measurement = "global"
kernels = profiler.insert_profiler(codegen_dir = "./build/codegen/host/src/",
                                   gdb_script_name = "./gdb_demo.sh",
                                   csv_file = "profile.csv",
                                   interactive = False,
                                   measurement = measurement)
kernels = profiler.insert_profiler(codegen_dir = "./build/codegen/host/src/",
                                   gdb_script_name = "./gdb_demo.sh",
                                   csv_file = "memory.csv",
                                   gdb_log_name= "memory.txt",
                                   interactive = False,
                                   measurement = "memory")
utils.make(device)
result_pulp = utils.gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
print("TEST: obtaining Diana output")
print(result_pulp)

if precision == 2:
    print("TEST: Verification for int2 is not supported on X86")
else:
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
perf_counters = profiler.process_profiler(measurement, kernels)
perf_counters = profiler.process_profiler("memory", kernels=None,
                                          log_file="memory.txt",
                                          csv_file="memory.csv")
