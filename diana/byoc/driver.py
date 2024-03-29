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
import argparse
import onnx
from tvm.relay.backend.contrib.soma_dory.onnx_transform import DianaOnnxIntegerize

import mlperf_tiny.relay_ds_cnn
import mlperf_tiny.relay_mobilenet
import mlperf_tiny.relay_resnet
import mlperf_tiny.relay_dae

import tvm.relay as relay


class Driver(ABC):
    """ Abstract base class for HTVM Driver code

    Classes that inherit from this class implement a common interface to drive
    compilation, profiling and running of HTVM-generated code.

    NOTE: Some of the methods can not be called consecutively.
    This is the order in which you can call them:
    1) tvm_compile
    2) (add_profiler)
    3) gcc_compile
    4) run
    5) (process_profile)

    :param mod: TVM IRModule for the network to be compiled
    :param params: dictionary of weights to be used in network
    :param build_dir: directory in which the code will be put, built and ran.
    :param byoc_path: directory that contains necessary files to import.
        Some files that can be imported include Makefiles, headers, and 
        libraries.
    """
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = "."):
        """Constructor method
        """
        self.model = TVMCModel(mod, params)
        self.build_dir = build_dir
        self.byoc_path = byoc_path

    @abstractmethod
    def tvm_compile(self, 
                    target: str = "c",
                    fusion: bool = False):
        """Compiles network to C code with TVM

        This is a wrapper method around TVMC that generates C code for the 
        network and C code that calls the network.
        All output is stored in self.build_dir

        :param target: target parameter passed to TVMC
        :param fusion: Enable/Disable operator fusion pass for TVM generated
            kernels
        """
        raise NotImplementedError()

    @abstractmethod
    def gcc_compile(self, gcc_opt: int = 3):
        """Compiles C code to self-contained binary with GCC
        
        This is a wrapper method around GCC that compiles the C code generated
        by the tvm_compile method in self.build_dir
        
        :param gcc_opt: gcc's optimization flag (e.g. -O3)

        NOTE: This method will fail if it is not called after tvm_compile
        """
        raise NotImplementedError()

    def add_profiler(self):
        """Adds profiling code to TVM-generated C code
        
        If implemented, this method adds profiling C-stubs to TVM-generated
        C-code. Profiling output is generated by the run method and also 
        put in self.build_dir.
        
        NOTE: This method will fail if it is not called after tvm_compile
        """
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        """Runs generated binary (and collects profiling information)
        
        This method runs the generated binary and collects profiling 
        information if the binary was gcc_compiled with added profilers from
        the add_profiler method.

        NOTE: This method will fail if it is not called after gcc_compile
        No profiler output is generated in self.build_dir if the C code was
        not compiled with profiling stubs from the add_profiler method

        :return: output of the network
        """
        raise NotImplementedError()

    def process_profile(self):
        """Post-processes profiling information
        
        If implemented, this method post-processes profiling information
        generated by run in self.build_dir by C code that has profiling stubs.
        """
        raise NotImplementedError()


class X86Driver(Driver):
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = "."):
        super(X86Driver, self).__init__(mod, params, build_dir, byoc_path)
        self.device = "x86"
        self.build_dir = self.build_dir / self.device
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)

    def tvm_compile(self, 
                    target: str ="c",
                    fusion: bool = False):
        utils.tvmc_compile_and_unpack(self.model,
                                      target=target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        utils.create_demo_file(self.model,
                               directory=self.build_dir,
                               use_printf=True)

    def gcc_compile(self, gcc_opt: int = 0):
        utils.adapt_gcc_opt(self.build_dir/"Makefile.x86", gcc_opt)
        utils.make(self.device, make_dir=self.build_dir)

    def run(self, gdb=True):
        if gdb:
            result_x86 = utils.gdb(device=self.device, binary="demo",
                                   gdb_script="gdb_demo_x86.sh",
                                   directory=self.build_dir)
            return result_x86
        else:
            # run binary and raise a assertion exception if the binary's return code
            # is different from zero
            utils.run_x86(self.build_dir/"demo")


class DianaDriver(Driver):
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = ".",
                 dory_path: pathlib.Path = "/dory"):
        super(DianaDriver, self).__init__(mod, params, build_dir, byoc_path)
        self.device = "pulp"
        self.build_dir = self.build_dir / self.device
        # Placeholders in case profiling code is added
        self.kernels = None
        self.measurement = None
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)
        utils.copy_dory_files(dory_path, self.build_dir)

    def tvm_compile(self,
                    target: str = "soma_dory, c",
                    fusion: bool = True,
                    indefinite: bool = False,
                    boot_analog: bool = False,
                    ):
        """Compiles network to C code with TVM

        This is a wrapper method around TVMC that generates C code for the 
        network and C code that calls the network.
        All output is stored in self.build_dir

        :param target: target parameter passed to TVMC
        :param fusion: Enable/Disable operator fusion pass for TVM generated
            kernels
        :param indefinite: put infinite loop around TVM network. Useful for
            power measurements.
        :param boot_analog: put analog core boot code in C wrapper before
            calling TVM generated code.
        """
        utils.tvmc_compile_and_unpack(self.model,
                                      target=target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        # This temporary file is generated in the previous function and is
        # used for processing the output of individual profiling data
        if "soma_dory" in target:
            shutil.copyfile("/tmp/macs_report.txt",self.build_dir/"macs_report.txt")
        utils.create_demo_file(self.model,
                               directory=self.build_dir,
                               boot_analog=boot_analog)

    def gcc_compile(self, gcc_opt: int = 3):
        utils.adapt_gcc_opt(self.build_dir/"Makefile.pulprt", gcc_opt)
        utils.make(self.device, make_dir=self.build_dir)

    def add_profiler(self, measurement):
        """Adds profiling code to TVM-generated C code
        
        This method adds profiling C-stubs to TVM-generated
        C-code. Profiling output is generated by the run method and also 
        put in self.build_dir.

        :param measurement: choose between "invidividual" and "global"
            "individual" measures individual kernel performance
            "global" measures the performance of the entire network
        
        NOTE: This method will fail if it is not called after tvm_compile
        NOTE: This method stores kernels and measurement attributes in the
            object which is needed later for process_profile
        """
        self.measurement = measurement
        self.kernels = profiler.insert_profiler(
                codegen_dir = self.build_dir/"codegen/host/src/",
                gdb_script_name = self.build_dir/"./gdb_demo.sh",
                csv_file = self.build_dir/"profile.csv",
                gdb_log_name = self.build_dir/"profile.txt",
                interactive = False,
                measurement = measurement)
        _ = profiler.insert_profiler(
                codegen_dir = self.build_dir/"/codegen/host/src/",
                gdb_script_name = self.build_dir/"gdb_demo.sh",
                csv_file = self.build_dir/"memory.csv",
                gdb_log_name= self.build_dir/"memory.txt",
                interactive = False,
                measurement = "memory")

    def process_profile(self):
        cycles = profiler.process_profiler(
                            measurement=self.measurement, 
                            kernels=self.kernels,
                            log_file=self.build_dir/"profile.txt",
                            csv_file=self.build_dir/"profile.csv",
                            macs_report = self.build_dir/"macs_report.txt")
        heap_usage = profiler.process_profiler(
                            measurement="memory", 
                            kernels=None,
                            log_file=self.build_dir/"memory.txt",
                            csv_file=self.build_dir/"memory.csv")
        size_dict = utils.size_pulp(binary=self.build_dir/
                                    "pulpissimo/demo/demo")
        print("\n")
        print("----- L2 STATIC MEMORY USAGE ----")
        for key, value in size_dict.items():
            print(f"{key:10}: {value:8,} Bytes   " +\
                    f"({round(float(value)/1000):4} kB)")
        print("\n")

    def run(self):
        result = utils.gdb(device=self.device, 
                           binary="pulpissimo/demo/demo",
                           gdb_script="gdb_demo.sh",
                           directory=self.build_dir)
        return result


def parse_onnx_input(filename: str):
    """Parse onnx model file and run high level relay transformations, specific to DIANA

    This function also checks if the parent directory of `filename` contains `.npy` files.
    If yes, these are parsed as well and stored in the params dict.
    These are later used in create_demo_file

    filename:   Path to the ONNX file

    Returns the parsed IR module and parameters dict
    """
    onnx_model = onnx.load(filename)
    ir_module, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
    # Diana quantlib specific interpretation pass
    ir_module = DianaOnnxIntegerize()(ir_module)

    # Check for .npy files. If yes, parse them and store them in params
    for fname in pathlib.Path(filename).parent.glob('*.npy'):
        # We add `g_` as prefix to avoid TVM from treating them as model constants
        # than can be constant-folded if their name matches a variable in the IR module
        param_name = 'g_' + fname.stem
        params[param_name] = np.load(fname)

    return ir_module, params


def driver(mod: tvm.ir.IRModule, 
           params: Dict[str, tvm.nd.array],
           run: bool = False, build_dir: pathlib.Path = "build",
           byoc_path: pathlib.Path = "."):
    """
    Compile (and run) a model for DIANA for testing purposes

    If the run argument is used, then it will also run the compiled model
    on a remote GDB instance over port 3333.
    Afterwards an x86 compiled model is used to compare outputs
    """
    # Create the model library format file and unpack
    d_diana = DianaDriver(mod, params, build_dir=build_dir,
                          byoc_path=byoc_path)
    d_diana.tvm_compile(fusion=True, target="soma_dory -layout_transform=0, c")
    d_diana.add_profiler(measurement="global")
    d_diana.gcc_compile(gcc_opt=3)
    # Make for DIANA
    if run:
        # Run the binary on DIANA
        output_pulp = d_diana.run()
        d_diana.process_profile()
        # Compile and run the network on x86
        d_x86 = X86Driver(mod, params, build_dir, byoc_path)
        d_x86.tvm_compile(fusion=False)
        d_x86.gcc_compile(gcc_opt=0)
        output_x86 = d_x86.run()
        # Compare X86 and DIANA outputs
        # Use allclose, not allequal (in case of floats)
        assert np.ma.allclose(output_x86,output_pulp)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTVM Command Line Driver")
    parser.add_argument('--target', dest='target',
                        help="Target string to pass onto TVMC, note that " + \
                             "'-device=arm_cpu' is appended to the string " +\
                             "later",
                        default="soma_dory, c")
    parser.add_argument('--device', dest='device',
                        choices = ("pulp", "x86"),
                        help="Device to make binary for (default 'pulp')",
                        default="pulp")
    parser.add_argument('--no-fusion', dest='fusion',
                        help="Set TVM's Relay Fusion pass' "+\
                             "maximum fusion depth to 0",
                        action='store_const', const=False,
                        default=True)
    parser.add_argument('--no-run', dest='run',
                        help="Do not run the binary after compilation",
                        action='store_const', const=False,
                        default=True)
    parser.add_argument('--gcc-opt', dest='gcc_opt',
                        choices = (0, 1, 2, 3), type=int,
                        help="Set the gcc optimization level",
                        default=3)
    parser.add_argument('--byoc_path', dest='byoc_path', type=pathlib.Path,
                        help="Set path to BYOC folder",
                        default=pathlib.Path("/tvm-fork/diana/byoc"))
    parser.add_argument('--build_dir', dest='build_dir', type=pathlib.Path,
                        help="Set output build directory",
                        default=pathlib.Path("/tmp"))
    parser.add_argument('--onnx', type=pathlib.Path,
                        help="Input onnx file from quantlib", default=None)

    # New group for PULP specific arguments
    pulp_group = parser.add_argument_group("Diana/pulp-specific arguments")
    pulp_group.add_argument('--dory_path', dest='dory_path', type=pathlib.Path,
                        help="Set path to DORY folder",
                        default=pathlib.Path("/dory"))
    pulp_group.add_argument('--profile', dest='measurement',
                        help="Insert PULP performance counters into "+\
                             "generated C code; for each individual kernel,"+\
                             "for the entire TVM artefact, or " +\
                             "don't insert performance counters (default)",
                        choices=("individual", "global", None),
                        default=None)
    pulp_group.add_argument('--boot_analog', dest='boot_analog',
                        help="Insert analog core boot code",
                        action='store_const', const=True,
                        default=False)
    pulp_group.add_argument('--indefinite', dest='indefinite',
                        help="Insert infinite loop around TVM generated code",
                        action='store_const', const=True,
                        default=False)

    # New group for rerunning specific experiments
    exp_group = parser.add_argument_group("Run HTVM MLPerf Tiny experiments"+\
                                    " (NOTE: This overrides some arguments)")
    exp_group.add_argument('--net', dest='network', type=str,
                           choices = ("ds_cnn", "mobilenet", "resnet","dae"), 
                           help="Choose which MLPerf Tiny(TM) network to run",
                           default=None)
    exp_group.add_argument('--conf', dest='configuration', type=str,
                           choices = ("cpu", "digital", "analog", "mixed"), 
                           help="Choose configuration for the network to run",
                           default="digital")
    args = parser.parse_args()

    # Running with --net overrides some options
    if args.onnx is not None:
        ir_module, params = parse_onnx_input(args.onnx)

    elif args.network is not None:
        # Defaults
        args.device = "pulp"
        args.target = "soma_dory -layout_transform=0, c"
        args.measurement = "global"
        args.fusion = True
        args.gcc_opt = 3
        weight_bits = 8
        args.boot_analog = False
        mixed = False
        add_layout_transforms = True
        # overriding defaults
        if args.configuration == "cpu":
            args.target = "c"
            add_layout_transforms = False
        if args.configuration == "analog":
            # Disable digital accelerator pattern matching
            args.target = "soma_dory -layout_transform=0 -disable_digital_acc=1, c"
            args.boot_analog = True
            weight_bits = 2
        if args.configuration == "mixed":
            args.boot_analog = True
            mixed = True
            weight_bits = 2
        if args.network == "ds_cnn":
            network_model = mlperf_tiny.relay_ds_cnn.create_model
        elif args.network == "mobilenet":
            network_model = mlperf_tiny.relay_mobilenet.create_model
        elif args.network == "resnet":
            network_model = mlperf_tiny.relay_resnet.create_model
        elif args.network == "dae":
            network_model = mlperf_tiny.relay_dae.create_model
        ir_module, params = network_model(
                weight_bits = weight_bits,
                add_layout_transforms = add_layout_transforms,
                mixed = mixed)
    else:
        raise ValueError("Either provide --onnx or --net to specify a network model")

    # Some options shouldn't be used together
    if args.device=="x86":
        if "soma_dory" in args.target:
            raise ValueError("Dory codegen can not be compiled for "+ \
                             "--device=\"x86\", only for --device=\"pulp\"")
        if args.measurement is not None:
            raise ValueError("Profiling is not available for "+\
                             "--device=\"x86\", only for --device=\"pulp\"")

    # Return string which identifies options
    def get_options_string(args: argparse.Namespace):
        fusion_name = "fused" if args.fusion else "unfused"
        target_name = "dory" if "soma_dory" in args.target else "c"
        options_string = f"{args.device}_{target_name}_{fusion_name}" + \
                   f"_O{args.gcc_opt}_{args.measurement}"
        # in case a default experiment is chosen
        if args.configuration is not None:
            options_string = args.configuration + "_" + options_string
        if args.network is not None:
            return args.network + "_" + options_string
        # otherwise
        else:
            return options_string

    # Drive compilation
    if args.device == "pulp":
        driver = DianaDriver(ir_module, params, 
                             args.build_dir / get_options_string(args), 
                             args.byoc_path, 
                             args.dory_path)
        driver.tvm_compile(target=args.target,
                           fusion=args.fusion,
                           indefinite=args.indefinite,
                           boot_analog=args.boot_analog)
    elif args.device == "x86":
        driver = X86Driver(ir_module, params, 
                           args.build_dir / get_options_string(args), 
                           args.byoc_path)
        driver.tvm_compile(target=args.target,
                           fusion=args.fusion)
    if args.measurement is not None:
        driver.add_profiler(measurement=args.measurement)
    driver.gcc_compile(gcc_opt=args.gcc_opt)
    if args.run:
        output_pulp = driver.run()
        if args.measurement is not None:
            driver.process_profile()
