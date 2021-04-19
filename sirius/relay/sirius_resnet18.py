from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
import tvm

resnet18_mod, resnet18_params = relay.testing.resnet.get_workload(num_layers=18)
print(resnet18_mod)

# build
with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize':True}):
    resnet18_lib = relay.build(resnet18_mod, "sirius", params=resnet18_params)

# path lib
file_name = "resnet.so"

# Getting module
module = resnet18_lib.get_lib()

# export library
module.export_library(file_name, workspace_dir="/tmp/")
