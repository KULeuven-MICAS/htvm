import tvm
import argparse
from tvm import relay
from tvm.driver.tvmc.model import TVMCModel
from tvm.relay.backend.contrib.soma_dory.onnx_transform import DianaOnnxIntegerize
import onnx

from utils import tvmc_compile_and_unpack

parser = argparse.ArgumentParser(description="Quantlib ONNX model compiler for DIANA")
parser.add_argument('onnxfile', help="ONNX model file from quantlib to compile", default="onnx/ResNet_QL_NOANNOTATION.onnx")
args = parser.parse_args()

# load onnx model
onnx_model = onnx.load(args.onnxfile)
mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)

# Diana quantlib specific interpretation pass
mod = DianaOnnxIntegerize()(mod)

model = TVMCModel(mod, params)
tvmc_compile_and_unpack(model, "soma_dory,c", fuse_layers=True)
