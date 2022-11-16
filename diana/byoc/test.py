import relay_resnet20
from tvm.driver.tvmc.model import TVMCModel
import utils
import numpy as np
import profiler
import subprocess
import pathlib
import shutil

import tvm.relay as relay


   

def run_network_x86(name, f_create_model):
    np.random.seed(0)
    precision = 8
    mod, params = f_create_model(precision, True)
    model = TVMCModel(mod, params)
    init_value = 1

    # int2 verification is not available on X86
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
    return result_x86

def run_network_diana(name, f_create_model, precision, mixed, pulp_target, measurement="global", result_x86=None):
    np.random.seed(0)
    mod, params = f_create_model(precision, True, mixed)
    model = TVMCModel(mod, params)
    init_value = 1
    verification = None
    # run on Diana
    print("TEST: compiling for Diana")
    device = "pulp"
    fusion = True
    utils.tvmc_compile_and_unpack(model, target=pulp_target, fuse_layers=fusion)
    # Add analog boot code in case of precision
    if precision == 2 or mixed:
        utils.create_demo_file(mod, init_value=init_value, boot_analog=True)
        utils.adapt_gcc_opt("Makefile.pulprt", 3)
    else:
        utils.create_demo_file(mod, init_value=init_value)
        utils.adapt_gcc_opt("Makefile.pulprt", 3)
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
                "heap_usage": None, 
                "size_dict": None}
    size_dict = utils.size_pulp("build/pulpissimo/demo/demo")
    if precision == 2 or mixed:
        log = pathlib.Path("demo.txt")
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        input("Please continue the measurement manually... And press enter")
        result_pulp = utils.get_gdb_output("demo.txt")
    else:
        try:
            result_pulp = utils.gdb(device, "build/pulpissimo/demo/demo", "gdb_demo.sh")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"name": name,
                    "compilation": True,
                    "run": False, 
                    "verification": None, 
                    "cycles": None, 
                    "heap_usage": None, 
                    "size_dict": size_dict}
            print("TEST: obtaining Diana output")
            print(result_pulp)

    if precision == 2 or mixed:
        print("TEST: Verification for int2 is not supported on X86")
    else:
        print("Final Results")
        print("=============")
        print("X86 output:")
        print(result_x86)
        print("Diana output:")
        print(result_pulp)
        # use allclose for floating point values
        try:
            if np.ma.allclose(result_x86,result_pulp):
                print("TEST: PASS")
                verification = True
            else:
                print("TEST: FAIL")
                verification = False
        except Exception as e:
            verification = None

    cycles = profiler.process_profiler(measurement, kernels)
    heap_usage = profiler.process_profiler("memory", kernels=None,
                                              log_file="memory.txt",
                                              csv_file="memory.csv")
    print(size_dict)
    return {"name": name,
            "compilation": True,
            "run": True, 
            "verification": verification, 
            "cycles": cycles, 
            "heap_usage": heap_usage, 
            "size_dict": size_dict}



def print_results(result_dict):
    if type(result_dict['cycles']) == int:
        print(f"\tcycles: {result_dict['cycles']:,} cyc")
    elif result_dict['cycles'] is None:
        print(f"\tcycles: {result_dict['cycles']}")
    else:
        result_dict['cycles'].pretty_print()
        print("\n")
        result_dict['cycles'].print_total_cycles()
    print(f"run: {result_dict['name']}")
    print(f"\tcompld: {result_dict['compilation']}")
    print(f"\trunned: {result_dict['run']}")
    print(f"\tverifd: {result_dict['verification']}")
    if result_dict['heap_usage'] is None:
        print(f"\tpeakl2: {result_dict['heap_usage']}")
    else:
        print(f"\tpeakl2: {result_dict['heap_usage'][0]:,} bytes")
        print(f"\t@endl2: {result_dict['heap_usage'][1]:,} bytes")
    if result_dict["compilation"] is None or result_dict["compilation"] == False:
        print(f"\tstatl2: {result_dict['size_dict']}")
    else:
        print(f"\tstatl2: {result_dict['size_dict']['total']:,} bytes")
        print(f"\tstatl2: {result_dict['size_dict']['total']:,} bytes")
        print(f"\t\tTEXT: {result_dict['size_dict']['text']:,} bytes")
        print(f"\t\tDATA: {result_dict['size_dict']['data']:,} bytes")
        print(f"\t\tBSS : {result_dict['size_dict']['bss']:,} bytes")


if __name__ == "__main__":
    import mlperf_tiny
    import mlperf_tiny.relay_dae
    import mlperf_tiny.relay_ds_cnn
    import mlperf_tiny.relay_mobilenet
    import mlperf_tiny.relay_resnet

    # Test settings
    network_under_test = mlperf_tiny.relay_dae
    precision = 2
    mixed = True
    measurement = "global"
    name = "relay_dae"
    setting = "HTVM_opt_mixed"
    experiment_name = pathlib.Path(f"{name}_{measurement}_{precision}_bits_{setting}")
    folder = pathlib.Path("results")
    exp_folder = folder / experiment_name
    if precision == 2:
        test_target ="soma_dory -layout_transform=0 -disable_digital_acc=1, c"
    else:
        test_target ="soma_dory -layout_transform=0, c"
    #test_target ="c"

    network_file = pathlib.Path(network_under_test.__file__)
    network_create = network_under_test.create_model
    try:
        exp_folder.mkdir(parents=True)
    except FileExistsError as e:
        print(e)
        print(f"You are about to overwrite {exp_folder}, are you sure?")
        response = input("Type 'yes' to proceed...")
        if response != "yes":
            print("Did not get 'yes', exiting...")
            exit(1)
        shutil.rmtree(exp_folder)
        exp_folder.mkdir(parents=True)

    if precision == 8:
        result_x86 = run_network_x86('manual_test_x86', network_create)
    else:
        result_x86 = None
    result_dict =  run_network_diana(f"final_testing", network_create, precision, mixed, test_target, measurement, result_x86)
    print_results(result_dict) 
    print(f"Copying all results to {exp_folder}")
    shutil.copyfile(network_file, exp_folder/network_file.name)
    shutil.copytree(pathlib.Path("build"), exp_folder/"build")
    shutil.copytree(pathlib.Path("dory"), exp_folder/"dory")
    shutil.copytree(pathlib.Path("src"), exp_folder/"src")
    shutil.copytree(pathlib.Path("include"), exp_folder/"include")
    shutil.copyfile(network_file, exp_folder/network_file.name)
    shutil.copyfile("Makefile.pulprt", exp_folder/"Makefile.pulprt")
    try:
        shutil.copyfile("demo_x86.txt",exp_folder/"demo_x86.txt")
        shutil.copyfile("demo.txt",exp_folder/"demo.txt")
        shutil.copyfile("memory.csv",exp_folder/"memory.csv")
        shutil.copyfile("profile.csv",exp_folder/"profile.csv")
        shutil.copyfile("macs.csv",exp_folder/"macs.csv")
        # Remove macs.csv after copy!
        pathlib.Path("macs.csv").unlink(missing_ok=True)
    except FileNotFoundError as e:
        print(e)
