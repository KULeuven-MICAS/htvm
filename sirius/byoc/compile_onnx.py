import tvm
from tvm import relay
from tvm.relay import transform
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime
import onnx

#byoc_target = 'soma'
byoc_target = 'soma_dory'

# load onnx model
onnx_model = onnx.load('model.onnx')
shape_dict = {'input': (1, 3, 10, 20)}

if byoc_target == 'soma':
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=False)
else:
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

# compile the model
model = TVMCModel(mod, params)
compile_model(tvmc_model=model,
              target=f"{byoc_target}, c",
              executor=Executor("aot",
                                {"interface-api": "c",
                                 "unpacked-api": 1}
                                ),
              runtime=Runtime("crt"),
              output_format="mlf",
              package_path="./build_soma/model.tar",
              pass_context_configs=['tir.disable_vectorize=1']
            )
