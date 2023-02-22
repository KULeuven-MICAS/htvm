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
    :param no_of_inputs: amount of inputs for the IR module.
        This information is necessary in for generating C wrapper code in
        build_dir/src/demo.c.
    """
    def __init__(self,
                 mod: tvm.ir.IRModule,
                 params: Dict[str, tvm.nd.array],
                 build_dir: pathlib.Path = "build",
                 byoc_path: pathlib.Path = ".",
                 no_of_inputs: int = 1):
        """Constructor method
        """
        self.model = TVMCModel(mod, params)
        self.build_dir = build_dir
        self.byoc_path = byoc_path
        self.no_of_inputs = no_of_inputs

    @abstractmethod
    def tvm_compile(self, 
                    fusion: bool = False,
                    init_value: int = 1):
        """Compiles network to C code with TVM

        This is a wrapper method around TVMC that generates C code for the 
        network and C code that calls the network.
        All output is stored in self.build_dir

        :param fusion: Enable/Disable operator fusion pass for TVM generated
            kernels
        :param init_value: input value set in calling wrapper
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
                 byoc_path: pathlib.Path = ".",
                 no_of_inputs: int = 1):
        super(X86Driver, self).__init__(mod, params, build_dir, byoc_path, no_of_inputs)
        self.device = "x86"
        self.build_dir = self.build_dir / self.device
        self.target = "c"
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)

    def tvm_compile(self, fusion: bool = False, init_value: int = 1):
        utils.tvmc_compile_and_unpack(self.model, target=self.target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        utils.create_demo_file(self.model.mod, init_value=init_value,
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
        # Placeholders in case profiling code is added
        self.kernels = None
        self.measurement = None
        utils.create_build_dir(self.byoc_path, self.build_dir, self.device)

    def tvm_compile(self, 
                    fusion: bool = True,
                    init_value: int = 1,
                    indefinite: bool = False,
                    boot_analog: bool = False,
                    ):
        """Compiles network to C code with TVM

        This is a wrapper method around TVMC that generates C code for the 
        network and C code that calls the network.
        All output is stored in self.build_dir

        :param fusion: Enable/Disable operator fusion pass for TVM generated
            kernels
        :param init_value: input value set in calling wrapper
        :param indefinite: put infinite loop around TVM network. Useful for
            power measurements.
        :param boot_analog: put analog core boot code in C wrapper before
            calling TVM generated code.
        """
        utils.tvmc_compile_and_unpack(self.model, target=self.target,
                                      fuse_layers=fusion,
                                      byoc_path=self.byoc_path,
                                      build_path=self.build_dir)
        # This temporary file is generated in the previous function and is
        # used for processing the output of individual profiling data
        shutil.copyfile("/tmp/macs_report.txt",self.build_dir/"macs_report.txt")
        utils.create_demo_file(self.model.mod, 
                               init_value=self.init_value,
                               no_of_inputs=self.no_of_inputs,
                               directory=self.build_dir)
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
                measurement = "measurement")
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
    d_diana.add_profiler(measurement="global")
    d_diana.gcc_compile(gcc_opt=3)
    # Make for DIANA
    if run:
        # Run the binary on DIANA
        output_pulp = d_diana.run()
        d_diana.process_profile()
        # Compile and run the network on x86
        d_x86 = X86Driver(mod, params, build_dir, byoc_path, no_of_inputs)
        d_x86.tvm_compile(fusion=False)
        d_x86.gcc_compile(gcc_opt=0)
        output_x86 = d_x86.run()
        # Compare X86 and DIANA outputs
        # Use allclose, not allequal (in case of floats)
        assert np.ma.allclose(output_x86,output_pulp)