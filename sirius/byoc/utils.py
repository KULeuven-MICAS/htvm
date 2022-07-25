import pathlib
import tarfile
import shutil
import os
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime



def tvmc_wrapper(model, target="soma, c", fuse_layers=True):
    ''' 
    Utility wrapper for TVMC that sets supported
    :param model: TVMC model that you wish to compile
    :param target: Can be "soma, c" if you want to offload all possible computations 
                   to accelerator, and can be "c" for golden model checking.
    '''
    # Check arguments
    assert((target == "soma, c") or (target == "c"))
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
    return
            
def tvmc_compile_and_unpack(model, target="soma, c", 
        fuse_layers=True, build_path="./build"):
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

