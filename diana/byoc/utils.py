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
                           kernel_size=w_value.shape[2:],
                           out_dtype=b_value.dtype)
    x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)
    else:
        x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')


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
    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)
    else:
        x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')
    return x, params


def relay_soma_add(input_tensor_a: relay.Var,
                   input_tensor_b: relay.Var,
                   layer_name: str,
                   shift_bits: int = 0):
    """
    Creates a relay element-wise-add op which is SOMA compatible
    :param input_tensor_a: relay.Var for input tensor A
    :param input_tensor_b: relay.Var for input tensor B
    :param layer_name: string that determines relay variable naming
    :shift_bits: int that sets amount of bits to shift right. Value must be between [0,31]
    """
    # define operations
    a = relay.op.cast(input_tensor_a, 'int32')
    b = relay.op.cast(input_tensor_b, 'int32')
    x = relay.op.add(a, b)
    x = relay.op.right_shift(x, relay.const(shift_bits))
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')

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
                            build_path: str = pathlib.Path("./build"),
                            byoc_path: str = ".",
                            device="pulp"):
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
    # Compile new model
    mlf_path = build_path / "model.tar"
    tvmc_wrapper(model, target, fuse_layers, mlf_path)
    # extract mlf file
    mlf = tarfile.TarFile(mlf_path)
    mlf.extractall(build_path)
    # remove the archive
    os.remove(mlf_path)


def create_build_dir(byoc_path: str = ".",
                     build_path: str = "./build",
                     device: str = "pulp"):
    """
    param byoc_path: path to import Makefiles and C dependencies from
    """
    build_path = pathlib.Path(build_path)
    byoc_path = pathlib.Path(byoc_path)
    # check if build folder exists
    if build_path.is_dir():
        # remove build folder and all contents
        shutil.rmtree(build_path)
        # make the build folder again
        build_path.mkdir(parents=True)
    if not build_path.is_dir():
        # If no build folder exists create one
        build_path.mkdir(parents=True)
    # Copy over other necessary files
    if device == "pulp":
        makefile_pulprt = pathlib.Path("Makefile.pulprt")
        shutil.copyfile(src=byoc_path / makefile_pulprt,
                        dst=build_path / makefile_pulprt)
    elif device == "x86":
        makefile_x86 = pathlib.Path("Makefile.x86")
        shutil.copyfile(src=byoc_path / makefile_x86,
                        dst=build_path / makefile_x86)
    else:
        raise NotImplementedError
    src_dir = pathlib.Path("src")
    include_dir = pathlib.Path("include")
    # Copy over src, include and dory folders
    shutil.copytree(src=byoc_path / src_dir,
                    dst=build_path / src_dir, dirs_exist_ok=True)
    shutil.copytree(src=byoc_path / include_dir,
                    dst=build_path / include_dir, dirs_exist_ok=True)


def copy_dory_files(dory_path: str = "/dory",
                    build_path: str = "./build"):
    """
    Function that copies dory library files on dory_path to a build_path
    """
    dory_hal_dir = pathlib.Path(dory_path) / "dory/Hardware_targets/"\
            / "Diana/Backend_Kernels/dory-hal"
    dory_utils_dir = pathlib.Path(dory_path) / "dory/Hardware_targets/"\
            / "Diana/Diana_TVM/Utils_files"
    dory_src_dir = pathlib.Path(build_path) / "dory/src"
    dory_src_dir.mkdir(parents=True)
    dory_inc_dir = pathlib.Path(build_path) / "dory/include"
    dory_inc_dir.mkdir(parents=True)
    hal_src_files= ["digital_conv_2d.c",
                   "analog_conv_2d.c",
                   "digital_element_wise_sum.c",
                   "digital_depthwise_conv_2d.c",
                   "digital_fully_connected.c",
                   "encoders_instruction_memory.c",
                   "utils.c"]
    hal_include_files = ["kernels.h",
                         "encoders_instruction_memory.h",
                         "utils.h"]
    utils_src_files = ["dory.c",
                       "mem_controller.c",]
    utils_include_files = ["dory.h",
                           "mem_controller.h"]
    for src in hal_src_files:
        shutil.copyfile(dory_hal_dir/"src"/src, dory_src_dir/src)
    for inc in hal_include_files:
        shutil.copyfile(dory_hal_dir/"include"/inc, dory_inc_dir/inc)
    for src in utils_src_files:
        shutil.copyfile(dory_utils_dir/src, dory_src_dir/src)
    for inc in utils_include_files:
        shutil.copyfile(dory_utils_dir/inc, dory_inc_dir/inc)


def create_demo_file(model: TVMCModel, directory: str = "build",
                     indefinite: bool = False,
                     boot_analog: bool = False,
                     use_printf: bool = False):
    '''
    Function that creates a demo file in which inputs and outputs of the
    right size are allocated and setup automatically.

    model:  TVMCModel containing both the IRModule and the parameters dict. If the parameters
            dict contains NDArray constants with names starting with `g_`, each constant is
            converted to a C-array and embedded in its own header file.
            If a constant for example is named `g_input` and the model's input is named `input`
            (same name but without the `g_`), than its header file is included in demo.c and
            the C-array is used as input for the model. If also `g_output` is available,
            the actual output of the model is asserted against `g_output`.
            This way, the numeric correctness of the compiled model can be verified.
            If no constants starting with `g_` are available in the params dict,
            the model's input is initialized with zeros and the output is not asserted.
    indefinite:     Run the model in a while(1) loop. This is useful for power measurement.
    boot_analog:    Insert code to boot the analog core.
    use_printf:     Print out assertion failures with printf. If False, nothing will be printed.

    Reference on how to obtain model input/output details:
    https://discuss.tvm.apache.org/t/how-to-get-the-input-and-output-of-relay-call-node/8743
    '''
    mod = model.mod
    params = model.params
    directory = pathlib.Path(directory)
    def get_c_type(dtype):
        if dtype == "int8":
            return "int8_t"
        elif dtype == "float32":
            return "float"
        else:
            raise NotImplementedError

    def gen_array_header(name: str, value: np.typing.NDArray):
        c_type = get_c_type(value.dtype)
        c_code = f"static const {c_type} {name}[] = {{"
        for v in value.flatten():
            c_code += f"{v},"
        c_code += '};\n'

        filename = f"model_{name}.h"
        with open(directory/"include"/filename, "w") as file:
            file.writelines(c_code)

        return filename

    # Before you can get the input and output types of a relay node
    # you first have to run the InferType Relay pass
    # otherwise checked_type will return a ValueError
    print("Creating demo file: Inferring shapes and types...")
    mod = relay.transform.InferType()(mod)
    # Assuming the first arguments are the user-supplied input
    # Convert from TVM runtime datatype to numpy array
    input_params = mod["main"].params
    # Assuming there is only output to this Relay IRMod
    # Convert from TVM runtime datatype to numpy array
    output_shape = np.array(mod["main"].checked_type.ret_type.shape)
    output_dtype = mod["main"].checked_type.ret_type.dtype
    create_demo_gdb_scripts(output_dtype, directory=directory)
    output_type_decl = get_c_type(output_dtype)
    if boot_analog:
        analog_boot_include = "#include <utils.h>\n"
        analog_boot_code = "boot_diana();"
    else:
        analog_boot_include = ""
        analog_boot_code = ""

    mallocs = ""
    array_header_includes = ""
    check_code = ""
    frees = ""
    sizes = ""
    call =   "  struct tvmgen_default_outputs outputs = { .output = output, };\n"
    call +=  "  struct tvmgen_default_inputs inputs = {\n"
    inits = f"  // Load input data\n"

    all_inputs_known = True
    for input_param in input_params:
        input_type_decl = get_c_type(input_param.type_annotation.dtype)
        input_shape = input_param.type_annotation.shape
        input_name = input_param.name_hint.replace(':', '_')
        sizes +=   f"  const uint32_t {input_name}_size = {np.prod(input_shape)};\n"
        mallocs += f"  {input_type_decl} *{input_name} = ({input_type_decl}*)malloc_wrapper({input_name}_size * sizeof({input_type_decl}));\n"
        inits +=   f"  for (uint32_t i = 0; i < {input_name}_size; i++){{\n"
        # check if input tensor has parameter values stored in params, if so, init with these values, otherwise initialize with zero
        params_input_name = 'g_' + input_name
        if params_input_name in params.keys():
            input_value = params[params_input_name]
            filename = gen_array_header(params_input_name, input_value)
            del params[params_input_name]
            array_header_includes += f'#include "{filename}"\n'
            assert np.prod(input_value.shape) == np.prod(input_shape)
            inits +=   f"    {input_name}[i] = {params_input_name}[i];\n"
        else:
            all_inputs_known = False
            inits +=   f"    {input_name}[i] = 0;\n"
        inits +=    "  }\n"
        call +=    f"    .{input_name} = {input_name},\n"
        frees +=   f"  free_wrapper({input_name});\n"

    # Add code for comparing output with expected output if all input data and expected output is known
    params_output_name = 'g_output'
    if params_output_name in params.keys() and all_inputs_known:
        output_value = params[params_output_name]
        filename = gen_array_header(params_output_name, output_value)
        del params[params_output_name]
        array_header_includes += f'#include "{filename}"\n'

        check_code_printf = r'      printf("Values at index %d differ: %f != %f\n", ' + \
                            f'i, {params_output_name}[i], output[i]);\n' if use_printf else ''
        check_code += \
        f'  for (uint32_t i = 0; i < output_size; i++) {{\n' + \
        f'    if ({params_output_name}[i] != output[i]) {{\n' + check_code_printf + \
        f'      status = 1;\n' + \
        f'    }}\n' + \
        f'  }}\n\n'

    # Generate other intermediate results headers if any. Those are not included in demo.c
    # but can be useful for manual debugging when the output is not as expected.
    for k, v in params.items():
        if k.startswith('g_'):
            gen_array_header(k, v)

    mallocs += f"  {output_type_decl} *output = ({output_type_decl}*)malloc_wrapper(output_size * sizeof({output_type_decl}));\n\n"
    frees +=    "  free_wrapper(output);\n"
    sizes +=   f"  uint32_t output_size = {np.prod(output_shape)};\n\n"
    call +=     "  };\n\n"

    # Now produce the final c code
    c_code = "#include <stdio.h>\n"  +\
             "#include <stdint.h>\n" +\
             "#include \"tvmgen_default.h\"\n" +\
             "#include <tvm_runtime.h>\n" +\
             "#include <malloc_wrapper.h>\n" +\
             "#include <gdb_anchor.h>\n" +\
    analog_boot_include +\
    array_header_includes +\
    "\n" +\
    "int main(int argc, char** argv) {\n" +\
    analog_boot_code +\
    "  // Sizes automatically added by utils.create_demo_file\n" +\
    sizes + \
    mallocs + \
    inits + "\n" +\
    call + \
    "  int32_t status = 0;\n" + \
    ("  while (status == 0){\n   " if indefinite else "") + \
    "  status = tvmgen_default_run(&inputs, &outputs);\n\n" + \
    ("}\n" if indefinite else "") + \
    "  gdb_anchor();\n" + \
    check_code + \
    frees + \
    "  if(status != 0){\n" +\
    "    abort();\n" +\
    "  }\n" +\
    "  return 0;\n" +\
    "}\n"

    with open(directory/ "src/demo.c", "w") as file:
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

def make(device: str = "pulp", make_dir: str = ".", verbose: bool = False):
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
    output = subprocess.run(["make", "-f", makefile,
                                         "all"], cwd=make_dir,
                                         check=True,
                                         stderr=subprocess.STDOUT,
                                         universal_newlines=True)
    if verbose:
        print(output)
    print(f"Make: Built for '{device}'")

def create_demo_gdb_scripts(dtype : str = "int8", directory: str = "."):
    def get_gdb_type(dtype):
        if dtype == "int8":
            return "/d"
        elif dtype == "float32":
            return "/f"
        else:
            raise NotImplementedError()
    directory = pathlib.Path(directory)
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
    f'set logging file {directory.resolve()}/demo_x86.txt\n'
    pulp = preamble +\
    'target remote localhost:3333\n' +\
    'load\n' +\
    'break gdb_anchor\n' +\
    'c\n' +\
    'n\n' +\
    'n\n' +\
    f'set logging file {directory.resolve()}/demo.txt\n'
    # These parts are common again
    common =\
    'set logging on\n' +\
    f'print {get_gdb_type(dtype)} *output@output_size\n' +\
    'set logging off\n'
    with open(directory / "gdb_demo_x86.sh", "w") as gdb_script:
        gdb_script.write(x86 + common)
        print(f"Made gdb_demo_x86.sh for {dtype}")
    with open(directory / "gdb_demo.sh", "w") as gdb_script:
        gdb_script.write(pulp + common)
        print(f"Made gdb_demo.sh for {dtype}")

def gdb(device: str, binary: str = None,
        directory: pathlib.Path = pathlib.Path("."),
        gdb_script: str = None,
        verbose : bool = False) -> np.typing.NDArray:
    """
    Calls gdb run (batch mode) for binary with gdb_script on specified device
    If verbose is set, output is printed

    returns the parsed gdb output in a numpy float array
    """
    directory = pathlib.Path(directory)
    def print_error(log_file, gdb_output):
        print(f"Could not open {log_file} -> gdb output was:")
        print("============================================")
        print(gdb_output)
    if device == "x86":
        log = directory / "demo_x86.txt"
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        if binary is None:
            binary = "demo"
        if gdb_script is None:
            gdb_script = "gdb_demo_x86.sh"
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        out = gdb_x86(directory/gdb_script, directory/binary, verbose)
        print("GDB: Run on x86 finished")
        try:
            result = get_gdb_output(log)
        except FileNotFoundError as e:
            print_error(log, out)
            return None
        return result
    elif device == "pulp":
        log = directory / "demo.txt"
        # Remove previous log before proceeding
        log.unlink(missing_ok=True)
        if binary is None:
            binary = "pulpissimo/demo/demo"
        if gdb_script is None:
            gdb_script = "gdb_demo.sh"
        print(f"GDB: Running '{gdb_script}' on '{device}'...")
        out = gdb_pulp(directory/gdb_script, directory/binary, verbose)
        print("GDB: Run on PULP finished")
        #try:
        result = get_gdb_output(log)
        #except FileNotFoundError as e:
        #    print_error(log, out)
        #    return None
        return result
    else:
        raise ValueError(f"Device: '{device}' not supported")


def gdb_x86(gdb_script: str, binary: str, verbose: bool = False) -> str:
    output = subprocess.check_output(["gdb", binary, "-x", gdb_script,
                                      "-batch"],
                                     stderr=subprocess.STDOUT,
                                     timeout=10,
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
        values = []
        for number in list_numbers:
            # NaNs in gdb are displayed as "nan(0x234...)"
            if "nan" in number:
                values.append(float("nan"))
            else:
                values.append(float(number))
    values_from_test = np.array(values, dtype="float")
    # Values are returned from GDB in one big one-dimensional tensor
    # Reshaping here such that it matches the output
    return values_from_test
