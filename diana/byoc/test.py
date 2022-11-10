import relay_resnet20
from tvm.driver.tvmc.model import TVMCModel
import utils
import numpy as np
import profiler
import subprocess

import tvm.relay as relay
#from relay_simple import create_model
#from relay_resnet20 import create_model


   

def run_network(name, mod, params, precision, pulp_target, measurement="global"):
    target = pulp_target
    model = TVMCModel(mod, params)
    init_value = 1
    verification = None

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
    #if manual_layout_transform:
    #    target = "soma_dory, c"
    #else:
    #    target = 'soma_dory -layout_transform=0, c'
    fusion = True
    utils.tvmc_compile_and_unpack(model, target=pulp_target, fuse_layers=fusion)
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
    _ = profiler.insert_profiler(codegen_dir = "./build/codegen/host/src/",
                                       gdb_script_name = "./gdb_demo.sh",
                                       csv_file = "memory.csv",
                                       gdb_log_name= "memory.txt",
                                       interactive = False,
                                       measurement = "memory")
    
    try:
        utils.make(device)
    except subprocess.CalledProcessError:
        return {"name": name,
                "compilation": False, 
                "run": None, 
                "verification": None, 
                "cycles": None, 
                "peak_l2_usage": None, 
                "size_dict": None}
    size_dict = utils.size_pulp("build/pulpissimo/demo/demo")
    try:
        result_pulp = utils.gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"name": name,
                "compilation": True,
                "run": False, 
                "verification": None, 
                "cycles": None, 
                "peak_l2_usage": None, 
                "size_dict": size_dict}
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
        # use allclose for floating point values
        if np.ma.allclose(result_x86,result_pulp):
            print("TEST: PASS")
            verification = True
        else:
            print("TEST: FAIL")
            verification = False
    cycles = profiler.process_profiler(measurement, kernels)
    peak_l2_usage = profiler.process_profiler("memory", kernels=None,
                                              log_file="memory.txt",
                                              csv_file="memory.csv")
    return {"name": name,
            "compilation": True,
            "run": True, 
            "verification": verification, 
            "cycles": cycles, 
            "peak_l2_usage": peak_l2_usage, 
            "size_dict": size_dict}
    #text = size_dict["text"]
    #bss = size_dict["bss"]
    #data = size_dict["data"]
    #total = size_dict["total"]
    #print("\n----- L2 STATIC MEMORY USAGE -----")
    #print(f"L2 TEXT : {text:12,} bytes ({100 * text/total:3.1f}%)")
    #print(f"L2 BSS  : {bss:12,} bytes ({100 * bss/total:3.1f}%)")
    #print(f"L2 DATA : {data:12,} bytes ({100 * data/total:3.1f}%)")
    #print(f"L2 TOTAL: {total:12,} bytes (100%)\n")
    #print(f"L2 rel. total static memory usage : {100 * total/2**19:3.1f}%\n")

if __name__ == "__main__":
    #from single_layer.relay_dense import create_model
    #from single_layer.relay_conv2d import create_model
    #from single_layer.relay_dw_conv2d import create_model
    #from mlperf_tiny.relay_dae import create_model
    #from mlperf_tiny.relay_ds_cnn import create_model
    from mlperf_tiny.relay_mobilenet import create_model
    #from mlperf_tiny.relay_resnet import create_model

    # for reproducability
    np.random.seed(0)

    # Set precision to 2 for triggering analog core
    precision = 8
    #precision = 2
    #precision = 32

    manual_layout_transform = True
    mod, params = create_model(precision, manual_layout_transform)
    #result_dict =  run_network(f"fail_malloc_{precision}_bits", mod, params, precision, "soma_dory, c")
    result_dict =  run_network(f"fail_malloc_{precision}_bits", mod, params, precision, "c")
    print(f"run: {result_dict['name']}")
    print(f"\tcompld: {result_dict['compilation']}")
    print(f"\trunned: {result_dict['run']}")
    print(f"\tverifd: {result_dict['verification']}")
    print(f"\tcycles: {result_dict['cycles']}")
    print(f"\tpeakl2: {result_dict['peak_l2_usage']}")
    print(f"\tstatl2: {result_dict['size_dict']}")
 
