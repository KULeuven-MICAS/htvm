import pathlib
import tarfile
import shutil
import os
import tvm
import tvm.relay as relay
import numpy as np
from tvm.driver.tvmc.compiler import compile_model
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend import Executor, Runtime

from typing import Tuple, Dict
import numpy.typing as npt

def relay_soma_conv2d(input_tensor: relay.Var, layer_name: str, 
                      weights_shape: Tuple[int], 
                      w_value: npt.NDArray[np.int8],
                      b_value: npt.NDArray[np.int32],
                      strides: Tuple[int] = (1,1),
                      act: bool = False, 
                      shift_bits: int = 0) -> Tuple[relay.Var,
                                              Dict[relay.Expr, tvm.nd.array]]:
    '''
    Creates a relay conv2d graph which is SOMA compatible
    This means it can be offloaded to the accelerator.
    :param input_tensor: relay.Var for input
    :param layer_name: string that determines relay variable naming
    :param weights_shape: tuple describing shape of weight tensor
    :param w_value: numpy int8 tensor that contains weight values
    :param b_value: numpy int32 tensor that contains bias values
    :param strides: tuple describing convolution stride (x,y)
    :param act: bool that toggles extra ReLU to be added (see below)
    :shift_bits: int that sets amount of bits to shift right
        value must be between [0,31]
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
    NOTE: The shape of the bias is hardcoded to use the first dimension
    of the weights_shape

    This function returns the relay expression for this graph,
    along with a parameter dictionary
    '''
    # define weights and bias variables
    weights_name = layer_name + '.weights'
    bias_name = layer_name + '.bias'
    conv_channels = weights_shape[0]
    # define relay input vars
    w = relay.var(weights_name, relay.TensorType(weights_shape, 'int8'))
    b = relay.var(bias_name, relay.TensorType((conv_channels,), 'int32'))
    # define weights and bias values in params
    params = {weights_name: tvm.nd.array(w_value), 
              bias_name: tvm.nd.array(b_value)}
    # define ops for a convolution on SOMA
    if (weights_shape[-2:] == (3,3)):
        padding = (1,1)
    elif (weights_shape[-2:] == (1,1)): 
        padding = (0,0)
    else:
        raise ValueError("only Fx=1,Fy=1 or Fx=3,Fy=3 are supported")
    if not ((strides == (1,1)) or (strides == (2,2))):
        raise ValueError("only strides (1,1) and (2,2) are supported")
    x = relay.qnn.op.conv2d(input_tensor, w, relay.const(0), relay.const(0),
                            relay.const(1.0), relay.const(1.0), 
                            weights_shape[-2:], channels=conv_channels, 
                            strides=strides,
                            padding=padding)
    x = relay.op.nn.bias_add(x, b)
    x = relay.op.right_shift(x, relay.const(shift_bits)) 
    x = relay.op.clip(x, a_min=-128, a_max=127)
    x = relay.op.cast(x, 'int8')
    # Optional: ReLU
    if act:
        x = relay.op.clip(x, a_min=0, a_max=127)

    return x, params


def tvmc_wrapper(model: TVMCModel, target: str = "soma_dory, c", 
                 fuse_layers: bool = True):
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
    assert((target == "soma_dory, c") or (target == "c"))
    # This has to be set by default to use the C runtime
    pass_context_configs = ['tir.disable_vectorize=1']
    if(fuse_layers == False):
        pass_context_configs.append('relay.FuseOps.max_depth=1')
    compile_model(tvmc_model=model,
                  target=target,
                  executor=Executor("aot",
                                    {"interface-api": "c",
                                     "unpacked-api": 1}
                                    ),
                  runtime=Runtime("crt"),
                  output_format="mlf",
                  package_path="./build/model.tar",
                  pass_context_configs=pass_context_configs,
                  )
            
def tvmc_compile_and_unpack(model: TVMCModel, target: str ="soma_dory, c",
        fuse_layers: bool = True, build_path: str = "./build"):
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
    # check if build folder exists
    path = pathlib.Path(build_path)
    if(path.is_dir()):
        # remove build folder and all contents
        shutil.rmtree(path)
        # make the build folder again
        path.mkdir()
    # Compile new model
    tvmc_wrapper(model, target, fuse_layers)
    mlf_path = pathlib.Path("./build/model.tar")
    # extract mlf file
    mlf = tarfile.TarFile(mlf_path)
    mlf.extractall(path)
    # remove the archive
    os.remove(mlf_path)

def create_demo_file(mod: tvm.ir.IRModule, path: str = "src/demo.c"):
    '''
    Function that creates a demo file in which inputs and outputs of the 
    right size are allocated and setup automatically. Based on:

    https://discuss.tvm.apache.org/t/
    how-to-get-the-input-and-output-of-relay-call-node/8743
    '''
    # Before you can get the input and output types of a relay node
    # you first have to run the InferType Relay pass
    # otherwise checked_type will return a ValueError
    print("Creating demo file: Inferring shapes and types...")
    mod = relay.transform.InferType()(mod)
    # Assuming the first argument is the user-supplied input
    # Convert from TVM runtime datatype to numpy array
    input_shape = np.array(mod["main"].checked_type.arg_types[0].shape)
    input_dtype = mod["main"].checked_type.arg_types[0].dtype
    # Assuming there is only output to this Relay IRMod
    # Convert from TVM runtime datatype to numpy array
    output_shape = np.array(mod["main"].checked_type.ret_type.shape)
    output_dtype = mod["main"].checked_type.ret_type.dtype
    print("Creating demo file: Inferred shapes:")
    print(f"\tinput ({input_dtype}):")
    print(f"\t {input_shape}")
    print(f"\toutput ({output_dtype}):")
    print(f"\t {output_shape}")

    c_code = \
    """
#include <stdio.h>
#include <stdint.h>
#include <tvm_runtime.h>
#include "tvmgen_default.h"

int abs(int v) {return v * ((v > 0) - (v < 0)); }

int main(int argc, char** argv) {
    tvm_workspace_t app_workspace;
    static uint8_t g_aot_memory[TVMGEN_DEFAULT_WORKSPACE_SIZE];
    StackMemoryManager_Init(&app_workspace, g_aot_memory, TVMGEN_DEFAULT_WORKSPACE_SIZE);
    // Sizes automatically added by utils.create_demo_file
    """ + \
    f"\tuint32_t input_size = {np.prod(input_shape)};\n" + \
    f"\tuint32_t output_size = {np.prod(output_shape)};\n" + \
    """
    int8_t *input = malloc(input_size * sizeof(int8_t));
    int8_t *output = malloc(output_size * sizeof(int8_t));
    // Fill first input with ones
    for (uint32_t i = 0; i < input_size; i++){
        input[i] = 1;
    }
    struct tvmgen_default_outputs outputs = {
    	.output = output,
    };
    struct tvmgen_default_inputs inputs = {
    	.input = input,
    };
    int32_t status = tvmgen_default_run(&inputs, &outputs);
    free(input);
    free(output);
    if(status != 0){
        abort();
    }
    return 0;
}
    """
    print(c_code)
    with open(path, "w") as file:
        file.writelines(c_code)

