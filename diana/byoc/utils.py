import pathlib
import tarfile
import shutil
import ctypes
import re
import os
import subprocess
import argparse
import tvm
import tvm.relay as relay
import numpy as np
from tvm.driver.tvmc.compiler import compile_model
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend import Executor, Runtime

from typing import Tuple, Dict, Optional, Union
import numpy.typing as npt


def numpy_to_array(np_arr: npt.NDArray, dtype: str):
    """ Convert a numpy array to a TVM array with datatype `dtype`.
    Although such a function exists in TVM, it does not support creating TVM arrays with dtypes
    that are not supported in numpy, like 'int4' or 'int2'.
    :param np_arr: the given numpy array
    :param dtype:  the resulting data type of the TVM array
    :return: the TVM array
    """
    assert np_arr.flags["C_CONTIGUOUS"]

    arr = tvm.nd.empty(np_arr.shape, dtype)
    data = np_arr.ctypes.data_as(ctypes.c_void_p)
    nbytes = ctypes.c_size_t(np_arr.size * np_arr.dtype.itemsize)
    tvm.nd.check_call(tvm.nd._LIB.TVMArrayCopyFromBytes(arr.handle, data, nbytes))

    return arr


def relay_soma_layout_transform(x, shape):
    """
    Creates a relay layout transform that reverses chunks of four bytes
    """

    x = relay.reshape(x, (np.prod(shape) // 4, 4))
    x = relay.reverse(x, axis=1)
    x = relay.reshape(x, shape)

    return x


def relay_soma_conv2d(input_tensor: relay.Var, layer_name: str,
                      w_value: tvm.nd.array,
                      b_value: tvm.nd.array,
                      strides: Tuple[int, ...] = (1, 1),
                      padding: Tuple[int, ...] = (0, 0),
                      groups: int = 1,
                      act: bool = False,
                      shift_bits: int = 0) -> Tuple[relay.Var,
                                                    Dict[relay.Expr,
                                                         tvm.nd.array]]:
    '''
    Creates a relay conv2d op which is SOMA compatible
    This means it can be offloaded to the accelerator.
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param w_value: int8 tensor that contains weight values
    :param b_value: int32 tensor that contains bias values
    :param strides: tuple describing convolution stride (x,y)
    :param padding: tuple describing convolution padding
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    :return: tuple that contains relay param dictionary and relay
        expr for the subgraph.

    The Relay code for one of these layers looks like this:
    ```
        %0 = qnn.conv2d(%input, %conv1.weights,...,out_dtype="int32");
        %1 = nn.bias_add(%0, %conv1.bias);
        %2 = right_shift(%1, 4);
        %3 = clip(%2, a_min=-128f, a_max=127f);
        %4 = cast(%3, dtype="int8");
    ```
    If `act` is set to `True` an additional ReLU-like clip is added
    ```
        %5 = clip(%4, a_min=0f, a_max=127f);
    ```

    This function returns the relay expression for this graph,
    along with a parameter dictionary
    '''
    # define weights and bias variables
    weights_name = layer_name + '.weights'
    bias_name = layer_name + '.bias'

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(w_value.shape, w_value.dtype))
    b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))

    # define weights and bias values in params
    params = {weights_name: w_value, bias_name: b_value}

    # define operations
    x = relay.op.nn.conv2d(input_tensor, w,
                           strides=strides,
                           padding=padding,
                           groups=groups,
                           out_dtype=b_value.dtype)
    x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')

    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)

    return x, params


def relay_soma_dense(input_tensor: relay.Var, layer_name: str,
                     w_value: tvm.nd.array,
                     b_value: tvm.nd.array,
                     act: bool = False,
                     shift_bits: int = 0):
    """
    Creates a relay dense op which is SOMA compatible
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param w_value: int8 tensor that contains weight values, must be of shape (num_inputs, num_outputs, 1, 1)
    :param b_value: int32 tensor that contains bias values
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define weights and bias variables
    weights_name = layer_name + '.weights'
    bias_name = layer_name + '.bias'

    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(w_value.shape, w_value.dtype))
    b = relay.var(bias_name, relay.TensorType(b_value.shape, b_value.dtype))

    # define weights and bias values in params
    params = {weights_name: w_value, bias_name: b_value}

    # define operations
    x = relay.op.nn.dense(input_tensor, w, out_dtype=b_value.dtype)
    x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')

    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)

    return x, params


def relay_soma_add(input_tensor_A: relay.Var, 
                   input_tensor_B: relay.Var, 
                   layer_name: str,
                   act: bool = False,
                   shift_bits: int = 0):
    """
    Creates a relay dense op which is SOMA compatible
    :param input_tensor_A: relay.Var for input tensor A
    :param input_tensor_B: relay.Var for input tensor A
    :param layer_name: string that determines relay variable naming
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define operations
    x = relay.op.add(input_tensor_A, input_tensor_B)
    x = relay.op.cast(x, 'int32')
    x = relay.op.right_shift(x, relay.const(shift_bits))
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')

    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)

    return x



def create_random_array(shape: Tuple[int, ...], dtype: str) -> tvm.nd.array:
    """
    Generate random interger weights with numpy and converts them to a TVMArray with requested dtype.
    :param shape: tuple of ints that indicates size of array
    :param dtype: datatype that indicates the data type of the array
    :return: array which was loaded or created

    NOTE: The random data is integer, uniformely distributed and ranges from
    minimum to maximum depending on the data type:
    E.g. in8 --> [-128, 127]
    """
    def get_dtype_range():
        try:
            dtype_min = np.iinfo(dtype).min
            dtype_max = np.iinfo(dtype).max
        except ValueError:
            range_map = {
                'int4': (-8, 7),
                'int2': (-1, 1)     # technically this should be (-2, 1), but we prefer to not use -2
            }
            try:
                dtype_min, dtype_max = range_map[dtype]
            except KeyError:
                raise ValueError(f"Creating an array of dtype {dtype} is not supported")

        return dtype_min, dtype_max

    dtype_min, dtype_max = get_dtype_range()
    np_dtype = dtype
    if dtype in ['int4', 'int2']:
        np_dtype = 'int8'
    np_array = np.random.randint(low=dtype_min, high=dtype_max+1,
                                 size=shape, dtype=np_dtype)
    return numpy_to_array(np_array, dtype)


def tvmc_wrapper(model: TVMCModel, target: str = "soma_dory, c",
                 fuse_layers: bool = True, 
                 package_path: pathlib.Path = pathlib.Path("model.tar")):
    '''
    Utility wrapper for TVMC that sets supported
    :param model: TVMC model that you wish to compile
    :param target: Can be "soma, c" if you want to offload all possible
        computations to accelerator, and can be "c" for golden model checking.
    :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
        if set to False. This tells relay to not fuse operations.
        This can be useful when debuggin the TVM-generated c code kernels.
    '''
    # Check arguments
    #assert ((target == "soma_dory, c") or (target == "c"))
    # Add -device=arm_cpu as default device for TVM C codegen
    # This will use the arm_cpu relay strategy as opposed to the x86 one.
    target += " -device=arm_cpu"
    # This has to be set by default to use the C runtime
    pass_context_configs = ['tir.disable_vectorize=1']
    if not fuse_layers:
        pass_context_configs.append('relay.FuseOps.max_depth=1')
    compile_model(tvmc_model=model,
                  target=target,
                  executor=Executor("aot",
                                    {"interface-api": "c",
                                     "unpacked-api": 1}
                                    ),
                  runtime=Runtime("crt"),
                  output_format="mlf",
                  package_path=package_path,
                  pass_context_configs=pass_context_configs,
                  )


def tvmc_compile_and_unpack(model: TVMCModel, target: str = "soma_dory, c",
                            fuse_layers: bool = True,
                            build_path: str = "./build"):
    '''
    Utility function that calls tvmc_wrapper and extracts output mlf
    (= TVM model library format) file.
    :param model: TVMC model that you wish to compile
    :param target: Can be "soma, c" if you want to offload all possible
        computations to accelerator, and can be "c" for golden model checking.
    :param fuse_layers: sets relay.FuseOps.max_depth parameter to 1
        if set to False. This tells relay to not fuse operations.
        This can be useful when debuggin the TVM-generated c code kernels.
    :param build_path: path to export mlf file output to
    '''
    path = pathlib.Path(build_path)
    # check if build folder exists
    if path.is_dir():
        # remove build folder and all contents
        shutil.rmtree(path)
        # make the build folder again
        path.mkdir()
    if not path.is_dir():
        # If no build folder exists create one
        path.mkdir()
    # Compile new model
    mlf_path = path / "model.tar"
    tvmc_wrapper(model, target, fuse_layers, mlf_path)
    # extract mlf file
    mlf = tarfile.TarFile(mlf_path)
    mlf.extractall(path)
    # remove the archive
    os.remove(mlf_path)


def create_demo_file(mod: tvm.ir.IRModule, path: str = "src/demo.c", 
                     init_value: int = 1, indefinite: bool = False, 
                     boot_analog: bool = False):
    '''
    Function that creates a demo file in which inputs and outputs of the
    right size are allocated and setup automatically. Based on:

    https://discuss.tvm.apache.org/t/
    how-to-get-the-input-and-output-of-relay-call-node/8743
    '''
    def get_c_type(dtype):
        if dtype == "int8":
            return "int8_t"
        elif dtype == "float32":
            return "float"
        else:
            raise NotImplementedError
    # Before you can get the input and output types of a relay node
    # you first have to run the InferType Relay pass
    # otherwise checked_type will return a ValueError
    print("Creating demo file: Inferring shapes and types...")
    mod = relay.transform.InferType()(mod)
    # Assuming the first argument is the user-supplied input
    # Convert from TVM runtime datatype to numpy array
    input_shape = np.array(mod["main"].checked_type.arg_types[0].shape)
    input_dtype = mod["main"].checked_type.arg_types[0].dtype
    type_decl_in = get_c_type(input_dtype)
    # Assuming there is only output to this Relay IRMod
    # Convert from TVM runtime datatype to numpy array
    output_shape = np.array(mod["main"].checked_type.ret_type.shape)
    output_dtype = mod["main"].checked_type.ret_type.dtype
    create_demo_gdb_scripts(output_dtype)
    type_decl_out = get_c_type(output_dtype)
    if boot_analog:
        analog_boot_include = "#include <utils.h>\n"
        analog_boot_code = "boot_diana();"
    else:
        analog_boot_include = ""
        analog_boot_code = ""
    print("Creating demo file: Inferred shapes:")
    print(f"\tinput ({input_dtype}):")
    print(f"\t {input_shape}")
    print(f"\toutput ({output_dtype}):")
    print(f"\t {output_shape}")
    malloc_statements = \
    f"    {type_decl_in} *input = ({type_decl_in}*)malloc_wrapper(input_size * sizeof({type_decl_in}));\n"+\
    f"    {type_decl_out} *output = ({type_decl_out}*)malloc_wrapper(output_size * sizeof({type_decl_out}));"
    free_statements = \
        """
        free_wrapper(input);
        free_wrapper(output);
        """
    c_code = \
        f"""#include <stdio.h>
#include <stdint.h>
#include "tvmgen_default.h"
#include <tvm_runtime.h>
#include <malloc_wrapper.h>
#include <gdb_anchor.h>\n""" +\
    analog_boot_include +\
    """
int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
    """ +\
    analog_boot_code +\
    """
    // Sizes automatically added by utils.create_demo_file\n""" + \
    f"    uint32_t input_size = {np.prod(input_shape)};\n" + \
    f"    uint32_t output_size = {np.prod(output_shape)};\n" + \
    malloc_statements + \
    """\n
    // Fill first input with ones
    for (uint32_t i = 0; i < input_size; i++){
    """ + \
    f"        input[i] = {init_value};\n" +\
    """
    }

    struct tvmgen_default_outputs outputs = {
        .output = output,
    };
    struct tvmgen_default_inputs inputs = {
        .input = input,
    };

    int32_t status = 0;
    """ + \
    ("while (status == 0){   " if indefinite else "") + \
    """
         status = tvmgen_default_run(&inputs, &outputs);
    """ + \
    ("}" if indefinite else "") + \
    """
    gdb_anchor();
    """ + \
        free_statements + \
        """
    if(status != 0){
        abort();
    }
    return 0;
}
"""
    with open(path, "w") as file:
        file.writelines(c_code)


def adapt_gcc_opt(makefile_path: str, opt_level: int):
    '''
    Adapts this line in a file to change OPT_LEVEL:
        OPT_LEVEL = 3
    typically used for makefiles.

    NOTE: Raises runtime error if no alterations were made
    '''
    regex = r"(OPT_LEVEL =) (\d)"
    subst = f"\\1 {opt_level}"
    with open(makefile_path, "r+") as makefile:
        makefile_string = makefile.read()
        replaced_string, subs = re.subn(regex, subst, makefile_string,
                                        0, re.MULTILINE)
        if subs != 1:
            raise RuntimeError("Could not alter makefile opt level")
        makefile.seek(0)
        makefile.write(replaced_string)
        makefile.truncate()
        print(f"Changed opt_level to {opt_level} @ {makefile_path}")

def make(device: str = "pulp", verbose: bool = False):
    '''
    Invokes make from the current directory in a new subprocess

    :param device: select which device to call make for, "x86" or "pulp"
    :param verbose: print make output
    '''
    print(f"Make: Invoking make for device: '{device}'")
    if device == "x86":
        makefile = "Makefile.x86"
    elif device == "pulp":
        makefile = "Makefile.pulprt"
    else:
        raise ValueError(f"Device: '{device}' not supported")
    output = subprocess.check_output(["make", "-f", makefile, 
                                         "clean", "all"],
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)
    if verbose:
        print(output)
    print(f"Make: Built for '{device}'")

def create_demo_gdb_scripts(dtype : str = "int8"):
    def get_gdb_type(dtype):
        if dtype == "int8":
            return "/d"
        elif dtype == "float32":
            return "/f"
        else:
            raise NotImplementedError()
    preamble =\
    'set print elements 0\n' +\
    'set print repeats 0\n' +\
    'set pagination off\n'
    # These parts are not common
    x86 = preamble +\
    'break gdb_anchor\n' +\
    'run\n' +\
    'n\n' +\
    'n\n' +\
    'n\n' +\
    'set logging file demo_x86.txt\n'
    pulp = preamble +\
    'target remote localhost:3333\n' +\
    'load\n' +\
    'break gdb_anchor\n' +\
    'c\n' +\
    'n\n' +\
    'n\n' +\
    'set logging file demo.txt\n'
    # These parts are common again
    common =\
    'set logging on\n' +\
    f'print {get_gdb_type(dtype)} *output@output_size\n' +\
    'set logging off\n'
    with open("gdb_demo_x86.sh", "w") as gdb_script:
        gdb_script.write(x86 + common)
        print(f"Made gdb_demo_x86.sh for {dtype}")
    with open("gdb_demo.sh", "w") as gdb_script:
        gdb_script.write(pulp + common)
        print(f"Made gdb_demo.sh for {dtype}")

def gdb(device: str, binary: str = None, gdb_script: str = None, 
        verbose : bool = False) -> np.typing.NDArray:
    """
    Calls gdb run (batch mode) for binary with gdb_script on specified device
    If verbose is set, output is printed

    returns the parsed gdb output in a numpy float array
    """
    def print_error(log_file, gdb_output):
        print(f"Could not open {log_file} -> gdb output was:")
        print("============================================")
        print(gdb_output)
    if device == "x86":
        log = pathlib.Path("demo_x86.txt")
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        binary = "build/demo" if binary is None else binary
        gdb_script = "gdb_demo_x86.sh" if gdb_script is None else gdb_script
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        out = gdb_x86(gdb_script, binary, verbose)
        print("GDB: Run on x86 finished")
        try:
            result = get_gdb_output("demo_x86.txt")
        except FileNotFoundError as e:
            print_error(log, out)
            return None
        return result
    elif device == "pulp":
        log = pathlib.Path("demo.txt")
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        binary = "build/pulpissimo/demo/demo" if binary is None else binary
        gdb_script = "gdb_demo.sh" if gdb_script is None else gdb_script
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        out = gdb_pulp(gdb_script, binary, verbose)
        print("GDB: Run on PULP finished")
        try:
            result = get_gdb_output(log)
        except FileNotFoundError as e:
            print_error(log, out)
            return None
        return result
    else:
        raise ValueError(f"Device: '{device}' not supported")


def gdb_x86(gdb_script: str, binary: str, verbose: bool = False) -> str:
    output = subprocess.check_output(["gdb", binary, "-x", gdb_script, 
                                      "-batch"],
                                     stderr=subprocess.STDOUT,
                                     timeout=3,
                                     universal_newlines=True) 
    if verbose:
        print(output)
    return output


def gdb_pulp(gdb_script: str, binary: str, verbose: bool = False) -> str: 
    riscv_gdb = "/pulp-riscv-gnu-toolchain/bin/riscv32-unknown-elf-gdb"
    """
    NOTE for some reason this program exits with zero even after errors?
    https://sourceware.org/bugzilla/show_bug.cgi?id=13000
    (Bug was fixed in 2018)
    """
    timeout=40
    #try:
    output = subprocess.check_output([riscv_gdb, binary, "-x", gdb_script,
                                          "-batch"],
                                         stderr=subprocess.STDOUT,
                                         timeout=timeout,
                                         universal_newlines=True) 
    #except subprocess.TimeoutExpired as e:
    #    print(f"GDB timed out after {timeout} seconds! --> Output:")
    #    print("==================================================")
    #    print(e.stdout.decode())
    #    return None
    if verbose:
        print(output)
    return output
    
def size_pulp(binary: str, verbose: bool = False) -> Dict[str,int]: 
    riscv_size = "/pulp-riscv-gnu-toolchain/bin/riscv32-unknown-elf-size"
    output = subprocess.check_output([riscv_size, binary],
                                     stderr=subprocess.STDOUT,
                                     universal_newlines=True) 
    if verbose:
        print(output)
    out = [int(match.group()) for match in re.finditer("(\d+)", output,
                                                       re.MULTILINE)]
    return {"text": out[0],
            "data": out[1],
            "bss": out[2],
            "total": out[3]}



def get_gdb_output(gdb_log_path="debug/gdb.txt"):
    """
    Following lines use the logging output of gdb to match test results with model results

    logging is set by:
        (gdb) set logging on       --> log to gdb.txt
        (gdb) print some_variable
        $1 = \032
        (gdb) set logging off

    In the code below we use regex to match an output created by gdb for an array:
        (gdb) print *my_array@array_size
        $2 = { 69, 420, ... , 42}  --> will be logged to gdb.txt

    After some string manipulation this array is converted to a numpy array.
    This array is checked for a complete match with np.ma.allequal()

    raises FileNotFoundError in case the log file can not be opened
    """
    with open(gdb_log_path) as log:
        data = ""
        for line in log.readlines():
            data += line.strip()
        # Find the right string in gdb log
        matcher = re.compile(r"{.*}",flags=re.DOTALL)
        result = matcher.search(data)
        string = result.group(0)
        # "{ ... , ... }" --> "... , ..."
        string = string.replace("{","")
        string = string.replace("}","")
        # makes a list of numbers in string format
        list_numbers = string.split(",")
        # convert strings to integers
        values = [float(number) for number in list_numbers]
    values_from_test = np.array(values, dtype="float")
    # Values are returned from GDB in one big one-dimensional tensor
    # Reshaping here such that it matches the output
    return values_from_test
    

def parse_cli_options() -> Tuple[argparse.Namespace, str]:
    '''
    Utility function that reads arguments from command line
    usage: see script_name.py -h
    '''
    parser = argparse.ArgumentParser(description="Utility argparser\
                                                  for example scripts")
    parser.add_argument('--target', dest='target',
                        choices=("soma_dory, c", "c"),
                        help="Target string to pass onto TVMC, '-device=arm_cpu' is added to the string later",
                        default="soma_dory, c")
    parser.add_argument('--device', dest='device',
                        choices = ("pulp", "x86"),
                        help="Device to make binary for (which makefile to call), (default 'pulp')",
                        default="pulp")
    parser.add_argument('--profile', dest='measurement',
                        help="Insert PULP performance counters into generated C code; for each individual kernel, for the entire TVM artefact, or don't insert performance counters (default)",
                        choices=("individual", "global", "power", None),
                        default=None)
    parser.add_argument('--interactive', dest='interactive',
                        action='store_const', const=True,
                        help="Wait for user input to have performed measurement to parse profiler results",
                        default=False)
    parser.add_argument('--no-fusion', dest='fusion',
                        help="Set TVM's Relay Fusion pass maximum fusion depth to 0",
                        action='store_const', const=False,
                        default=True)
    parser.add_argument('--manual-layout-transform',
                        help="Insert hand-tuned layout transform locations rather than automatically",
                        action='store_true')
    parser.add_argument('--weight-bits', dest='weight_bits', type=int,
                        help="Number of bits per weight. This affects the selection of the digital/analog core",
                        choices=(8, 2),
                        default=8)
    parser.add_argument('--gcc-opt', dest='gcc_opt',
                        choices = (0, 1, 2, 3), type=int,
                        help="Set the gcc optimization level in pulprt makefile, (default Makefile.pulprt)",
                        default=3)
    parser.add_argument('--makefile', dest='makefile',
            help="Set different path for pulprt makefile (default for device = x86: Makefile.x86, default for  device = pulp: Makefile.pulprt")
    args = parser.parse_args()
    # Some options shouldn't be used together
    if args.device == "x86":
        if "soma_dory" in args.target:
            raise ValueError("Dory codegen can not be compiled for --device=\"x86\", only for --device=\"pulp\"")
        if args.measurement is not None:
            raise ValueError("Profiling is not available for --device=\"x86\", only for --device=\"pulp\"")
    # Set default makefile if no other is specified
    if args.makefile is None:
        if args.device == "pulp":
            args.makefile = "Makefile.pulprt"
        elif args.device == "x86":
            args.makefile = "Makefile.x86"
    adapt_gcc_opt(args.makefile, args.gcc_opt)
    # Return string which identifies options
    def get_options_string(args: argparse.Namespace):
        fusion_name = "fused" if args.fusion else "unfused"
        target_name = "dory" if args.target == "soma_dory, c" else "c"
        options_string = f"{args.device}_{target_name}_{fusion_name}" + \
                   f"_O{args.gcc_opt}_{args.measurement}"
        return options_string

    return args, get_options_string(args)
