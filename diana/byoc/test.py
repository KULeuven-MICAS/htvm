import numpy as np
import mlperf_tiny.relay_ds_cnn
import mlperf_tiny.relay_mobilenet
import mlperf_tiny.relay_resnet
import mlperf_tiny.relay_dae
import os
import pytest
import tvm
import tvm.relay as relay

from tvm.driver.tvmc.model import TVMCModel
from typing import Dict
import driver


def get_test_params():
    """
    This function is used to generate test parameter combinations for
    * Conv2D layers
    * DWConv2D layers

    Why not use pytest.mark.parametrize?
    --> this generates very long filenames with the tmp_path fixture
    --> long paths end up in the debug section of the output binary
    --> pulp RISC-V GDB will crash :(

    This method has some upsides:
    + Conv2D and DWConv2D use the same test params
    + GDB doesn't crash :)

    It has a very obvious downside:
    - Testnames are now very cryptic e.g. "id3, id4" :(
    """
    import itertools
    weight_bits = [8]
    act = [False, True]
    strides = [(1, 1), (2, 2)]
    kernel_and_padding = [[[7, 7], (3, 3)],
                          [[5, 5], (2, 2)], 
                          [[3, 3], (1, 1)], 
                          [[1, 1], (0, 0)],
                          [[7, 5], (3, 2)]]
    combination = [weight_bits, act, strides, kernel_and_padding]
    test_params = list(itertools.product(*combination))
    test_ids = ["id" + str(i) for i in range(len(test_params))]
    return test_params, test_ids


test_params, test_ids = get_test_params()
@pytest.mark.parametrize("test_params", test_params, ids=test_ids)
def test_conv2d(run, test_params, tmp_path):
    weight_bits, act, strides, kernel_and_padding = test_params
    import single_layer.relay_conv2d 
    # Set random seed for reproducible testing
    np.random.seed(0)
    kernel_size = kernel_and_padding[0]
    padding = kernel_and_padding[1]
    ir_module, params = single_layer.relay_conv2d.create_model(
        weights_shape = tuple([32, 32] + kernel_size),
        weight_bits = weight_bits,
        act = act,
        padding = padding,
        strides = strides,
        shift_bits = 4
            )
    # Run the test
    driver.driver(ir_module, params, run, tmp_path)


@pytest.mark.parametrize("kernel_size, padding", [((1, 1), (0, 0)), ((3, 3), (1, 1)), ((5, 5), (2, 2))])
def test_conv2d_reproduce_bug(run, tmp_path, kernel_size, padding):
    """Test to reproduce bug digital accelerator. Investigation needed.
    """
    import utils
    import single_layer.relay_conv2d

    # Set random seed for reproducible testing
    np.random.seed(0)

    ir_module, params = single_layer.relay_conv2d.create_model(
        input_shape = (1, 3, 32, 32),
        weights_shape = tuple([16, 3] + list(kernel_size)),
        weight_bits = 8,
        act = False,
        padding = padding,
        strides = (1, 1),
        shift_bits = 5
            )

    # set bias to zero (or small numbers are also fine), which is the trick that uncovered the bug
    params['conv1.bias'] = utils.numpy_to_array(np.zeros(16, dtype='int32'), 'int32')

    # load some known input, not all the same number
    input_data = np.load('test_data/export_resnet8/input.npy')
    params['g_input_0'] = (input_data * 64).astype('int8')

    # Run the layer on diana and x86 and compare outputs
    driver.driver(ir_module, params, run, tmp_path)


@pytest.mark.parametrize("test_params", test_params, ids=test_ids)
def test_dw_conv2d(run, test_params, tmp_path):
    weight_bits, act, strides, kernel_and_padding = test_params
    import single_layer.relay_dw_conv2d 
    # Set random seed for reproducible testing
    np.random.seed(0)
    kernel_size = kernel_and_padding[0]
    padding = kernel_and_padding[1]
    ir_module, params = single_layer.relay_dw_conv2d.create_model(
        weights_shape = tuple([32, 1] + kernel_size),
        weight_bits = weight_bits,
        act = act,
        padding = padding,
        strides = strides,
        shift_bits = 4
            )
    # Run the test
    driver.driver(ir_module, params, run, tmp_path)


@pytest.mark.parametrize("weight_bits", [8], ids = ["digital"])
@pytest.mark.parametrize("act", [False, True], ids = ["no_relu", "relu"])
def test_dense(run, weight_bits, act, tmp_path):
    import single_layer.relay_dense
    # Set random seed for reproducible testing
    np.random.seed(0)
    ir_module, params = single_layer.relay_dense.create_model(
        weight_bits = weight_bits,
        act = act,
        shift_bits = 4
            )
    # Run the test
    driver.driver(ir_module, params, run, tmp_path)


def test_add(run, tmp_path):
    import single_layer.relay_add
    # Set random seed for reproducible testing
    np.random.seed(0)
    ir_module, params = single_layer.relay_add.create_model(
        shift_bits = 4
            )
    # Run the test
    driver.driver(ir_module, params, run, tmp_path)

    
def run_full_network(run, directory, network):
    np.random.seed(0)
    ir_module, params = network(
        weight_bits = 8,
        # Disable manually inserted layout transforms
        add_layout_transforms = False,
        mixed = False)
    driver.driver(ir_module, params, run, directory)


###
#   Full network tests of models defined in relay.
#   If run is False, only compilation for DIANA is attempted.
#   If run is True, the model is also ran on DIANA and its output is compared to
#   the output of the same model, ran on x86.
###


def test_mlperf_tiny_ds_cnn(run, tmp_path):
    run_full_network(run, tmp_path, mlperf_tiny.relay_ds_cnn.create_model)


def test_mlperf_tiny_resnet(run, tmp_path):
    run_full_network(run, tmp_path, mlperf_tiny.relay_resnet.create_model)


def test_mlperf_tiny_mobilenet(run, tmp_path):
    run_full_network(run, tmp_path, mlperf_tiny.relay_mobilenet.create_model)


def test_mlperf_tiny_dae(run, tmp_path):
    run_full_network(run, tmp_path, mlperf_tiny.relay_dae.create_model)


###
#   Full network tests on x86 of models defined in ONNX files, generated by dianaquantlib.
#   IMPORTANT: these tests require git lfs checkout of the `test_data` folder in order
#   to install the .onnx test files used as input.
###


def run_onnx_x86(onnx_file: str, directory: str):
    """Compile and run an ONNX model on x86
    """
    # ensure .npy files are present to get numeric checking too
    onnx_dir = os.path.dirname(onnx_file)
    assert os.path.exists(os.path.join(onnx_dir, 'input.npy'))
    assert os.path.exists(os.path.join(onnx_dir, 'output.npy'))

    mod, params = driver.parse_onnx_input(onnx_file)
    d = driver.X86Driver(mod, params, directory)
    d.tvm_compile()
    d.gcc_compile(gcc_opt=0)
    d.run(gdb=False)   # raises exception on failure


def test_mlperf_tiny_ds_cnn_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_dscnn/DSCNN_QL_NOANNOTATION.onnx', tmp_path)


def test_mlperf_tiny_resnet_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_resnet8/ResNet_QL_NOANNOTATION.onnx', tmp_path)


def test_mlperf_tiny_mobilenet_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_mobilenetv1/MobileNet_QL_NOANNOTATION.onnx', tmp_path)


def test_mlperf_tiny_dae_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_dae/DAE_QL_NOANNOTATION.onnx', tmp_path)


def test_cifar10_resnet20_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_resnet20/ResNet_QL_NOANNOTATION.onnx', tmp_path)


def test_cifar10_mobilenetv2_onnx_x86(tmp_path):
    run_onnx_x86('test_data/export_mobilenetv2/MobileNetV2_QL_NOANNOTATION.onnx', tmp_path)
