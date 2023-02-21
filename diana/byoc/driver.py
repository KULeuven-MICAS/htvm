from tvm.driver.tvmc.model import TVMCModel
from typing import Dict
from abc import ABC, abstractmethod
import tvm
import utils
import numpy as np
import profiler
import subprocess
import pathlib
import shutil

import mlperf_tiny.relay_ds_cnn
import mlperf_tiny.relay_mobilenet
import mlperf_tiny.relay_resnet
import mlperf_tiny.relay_dae

import tvm.relay as relay


class Driver(ABC):
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = ".",
                 no_of_inputs: int = 1):
        self.model = TVMCModel(mod, params)
        self.build_dir = build_dir
        self.byoc_path = byoc_path
        self.no_of_inputs = no_of_inputs

    @abstractmethod
    def tvm_compile(self, fusion: bool = False):
        raise NotImplementedError()

    @abstractmethod
    def gcc_compile(self, gcc_opt: int = 3):
        raise NotImplementedError()

    def add_profiler(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def profile(self):
        raise NotImplementedError()


class X86Driver(Driver):
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = ".",
                 no_of_inputs: int = 1):
        super(X86Driver, self).__init__(mod, params, build_dir, byoc_path, no_of_inputs)
        self.device = "x86"
        self.build_dir = self.build_dir / self.device
        self.target = "c"
        self.init_value = 1
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)

    def tvm_compile(self, fusion: bool = False):
        utils.tvmc_compile_and_unpack(self.model, target=self.target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        utils.create_demo_file(self.model.mod, init_value=self.init_value,
                               no_of_inputs=self.no_of_inputs,
                               directory=self.build_dir)

    def gcc_compile(self, gcc_opt: int = 0):
        utils.adapt_gcc_opt(self.build_dir/"Makefile.x86", gcc_opt)
        utils.make(self.device, make_dir=self.build_dir)

    def run(self):
        result_x86 = utils.gdb(device=self.device, binary="demo",
                               gdb_script="gdb_demo_x86.sh",
                               directory=self.build_dir)
        return result_x86

class DianaDriver(Driver):
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = ".",
                 no_of_inputs: int = 1):
        super(DianaDriver, self).__init__(mod, params, build_dir, byoc_path, no_of_inputs)
        self.device = "pulp"
        self.build_dir = self.build_dir / self.device
        # TODO: move -requant_transform somewhere else?
        self.target="soma_dory -requant_transform=0, c"
        self.init_value = 1
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)

    def tvm_compile(self, fusion: bool = True):
        utils.tvmc_compile_and_unpack(self.model, target=self.target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        utils.create_demo_file(self.model.mod, init_value=self.init_value,
                               no_of_inputs=self.no_of_inputs,
                               directory=self.build_dir)
    def gcc_compile(self, gcc_opt: int = 3):
        utils.adapt_gcc_opt(self.build_dir/"Makefile.pulprt", gcc_opt)
        utils.make(self.device, make_dir=self.build_dir)

    def run(self):
        result = utils.gdb(device=self.device, 
                           binary="pulpissimo/demo/demo",
                           gdb_script="gdb_demo.sh",
                           directory=self.build_dir)
        return result


def driver(mod: tvm.ir.IRModule, 
           params: Dict[str, tvm.nd.array],
           run: bool = False, build_dir: pathlib.Path = "build",
           byoc_path: pathlib.Path = ".",
           no_of_inputs: int = 1):
    """
    Compile (and run) a model for DIANA for testing purposes

    If the run argument is used, then it will also run the compiled model
    on a remote GDB instance over port 3333.
    Afterwards an x86 compiled model is used to compare outputs
    """
    # Create a TVMCModel
    model = TVMCModel(mod, params)
    # Create the model library format file and unpack
    d_diana = DianaDriver(mod, params, build_dir=build_dir,
                          byoc_path=byoc_path, no_of_inputs=no_of_inputs)
    d_diana.tvm_compile(fusion=True)
    d_diana.gcc_compile(gcc_opt=3)
    # Make for DIANA
    if run:
        # Run the binary on DIANA
        output_pulp = d_diana.run()
        # Compile and run the network on x86
        d_x86 = X86Driver(mod, params, build_dir, byoc_path, no_of_inputs)
        d_x86.tvm_compile(fusion=False)
        d_x86.gcc_compile(gcc_opt=0)
        output_x86 = d_x86.run()
        # Compare X86 and DIANA outputs
        # Use allclose, not allequal (in case of floats)
        assert np.ma.allclose(output_x86,output_pulp)



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
    # Test settings
    network_under_test = mlperf_tiny.relay_resnet
    precision = 2
    mixed = False
    measurement = "global"
    name = "relay_resnet_no_ews"
    setting = "HTVM_opt_no_dma"
    experiment_name = pathlib.Path(f"{name}_{measurement}_{precision}_bits_{setting}")
    folder = pathlib.Path("results")
    exp_folder = folder / experiment_name
    if precision == 2 and not mixed:
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
        #result_x86 = run_network_x86('manual_test_x86', network_create)
        result_x86 = None
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
