#*************************************************************************
# (LLVM) In this Relay example I try to implement a simple matrix addition
#*************************************************************************

import tvm
import tvm.relay as relay
import tvm.relay.op.contrib.soma as soma
import numpy as np
import tvm.driver.tvmc as tvmc

#from tvm.contrib import graph_runtime
#from aot_tvm import tvm_compile
import os

# create tensor variables with dimensions (batch,channels,x,y)
# tensor_shape = (1,1,8,8)
# Or create a 2D-matrix
tensor_shape = (16, 16, 16)
data_type = "int8"

# Construct the variables --> tvm.relay.Var type
a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))

# Then we tell it to add the two variables --> tvm.relay.Expr type
sum_expr = relay.add(a, b)

# Now create an IRModule from the tvm.relay.Expr file
module = tvm.ir.IRModule()
module = module.from_expr(sum_expr)

# As in documentation: https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm#bring-dnnl-to-tvm-annotation-rules

#module = relay.transform.MergeComposite(soma.pattern_table())(module)
#module = relay.transform.AnnotateTarget(["soma"])(module)
#print(module)
#module = relay.transform.AnnotateTarget(["c"])(module)
#module = relay.transform.MergeCompilerRegions()(module)
#print(module)
#module = relay.transform.PartitionGraph()(module)
#print(module)

# Define a target for compilation
target = tvm.target.Target("c")

# Optimize (?)  and build the relay code:
print(module)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(module, target)

## Old way of compiling
#
#build_directory = "/tmp/"
#
library = lib.get_lib()
json = lib.get_graph_json()
new_params = lib.get_params()
#
#try:
#    library.export_library(
#        "soma.so", workspace_dir=build_directory
#    )
#except Exception as e:
#    print(e)
#    pass
#

# New way of compiling (with TVMC)
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime


model = TVMCModel(module, new_params)
compile_model(tvmc_model=model,
              target="c",
              executor=Executor("aot",
                                {"interface-api": "c",
                                 "unpacked-api": 1}
                                ),
              runtime=Runtime("crt"),
              output_format="mlf",
              package_path="./model.tar",
              pass_context_configs=['tir.disable_vectorize=1']
            )



#tvm_compile(
#    json,
#    new_params,
#    os.path.join(build_directory, "soma_main.c"),
#    os.path.join(build_directory, "soma.h"),
#    verbose=True,
#    get_per_node=False,
#    add_perf_counter=True,
#    perf_type="c",
#)

