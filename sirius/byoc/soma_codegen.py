import tvm
import tvm.relay as relay
import tvm.relay.op.contrib.soma as soma
import numpy as np
import tvm.driver.tvmc as tvmc
import pathlib
from tvm.driver.tvmc.model import TVMCModel
from tvm.driver.tvmc.compiler import compile_model
from tvm.relay.backend import Executor, Runtime

def create_inputs_c_header(section_name, input_a, input_b, w, h, c):
    r"""
    Creates a headerfile which contains the inputs of the neural network.
    Based off of: https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_ethosu.html#sphx-glr-how-to-work-with-microtvm-micro-ethosu-py

    Parameters:
    -----------
    section_name: str
        name of the section to include as attribute in final ELF file

    input_a: numpy array with 3 dimensions
    input_b: numpy array with 3 dimensions
        refers to input a and b defined in main above
    """ 
    file_path = pathlib.Path(f"include/inputs.h").resolve()
    with open(file_path, "w") as header_file:
        header_file.write(
                "// inputs.h : automatically generated header file by soma_codegen.py\n" +
                f"const size_t tensor_b_size = {input_b.size};\n" 
                )

        def write_tensor(input_tensor, name):
            header_file.write(f"const size_t {name}_size = {input_tensor.size};\n")
            header_file.write(f"const size_t {name}_w = {w};\n")
            header_file.write(f"const size_t {name}_h = {h};\n")
            header_file.write(f"const size_t {name}_c = {c};\n\n")
            data_hexstr_input = input_tensor.tobytes().hex()
            header_file.write(f"int8_t {name}[] __attribute__((section(\"{section_name}\"))) = "+"\"")
            for i in range(0, len(data_hexstr_input), 2):
                header_file.write(f"\\x{data_hexstr_input[i:i+2]}")
            header_file.write("\";\n\n")
        
        write_tensor(input_a, "a")
        write_tensor(input_b, "b")


def input_maker(w, h, c):
    # Copied from old demo.c
    # Construct as flat array
    input_a = np.empty(shape=w*h*c, dtype="int8")
    input_b = np.empty(shape=w*h*c, dtype="int8")
    for i in range(w):
        for j in range(h):
            for k in range(c):
                position = i*h*c + j*c + k;
                input_a[position] = i*j + k 
                input_b[position] = i + j - 2*k
    return input_a, input_b
 
if __name__ == "__main__":
    tensor_shape = (3, 15, 17)
    data_type = "int8"
    # Construct the variables --> tvm.relay.Var type
    a = relay.var("a", tvm.relay.TensorType(tensor_shape, data_type))
    b = relay.var("b", tvm.relay.TensorType(tensor_shape, data_type))
    # Then we tell it to add the two variables --> tvm.relay.Expr type
    sum_expr = relay.add(a, b)
    # Now create an IRModule from the tvm.relay.Expr file
    module = tvm.ir.IRModule()
    module = module.from_expr(sum_expr)

    # As in documentation:
    # https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm#bring-dnnl-to-tvm-annotation-rules

    # module = relay.transform.MergeComposite(soma.pattern_table())(module)
    module = relay.transform.AnnotateTarget(["soma"])(module)
    module = relay.transform.MergeCompilerRegions()(module)
    module = relay.transform.PartitionGraph()(module)

    # Define a target for compilation
    target = tvm.target.Target("c")

    # Optimize and build the relay code:
    print("compiling this relay node:")
    print(module)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(module, target)

    library = lib.get_lib()
    json = lib.get_graph_json()
    new_params = lib.get_params()
    # New way of compiling (with TVMC)

    print("creating inputs:")
    w, h, c = tensor_shape
    input_a, input_b = input_maker(w, h, c)
    create_inputs_c_header("input_section", input_a, input_b, w, h, c)

    print("compiling TVMC model:")
    model = TVMCModel(module, new_params)
    compile_model(tvmc_model=model,
                  target="soma, c",
                  executor=Executor("aot",
                                    {"interface-api": "c",
                                     "unpacked-api": 1}
                                    ),
                  runtime=Runtime("crt"),
                  output_format="mlf",
                  package_path="./model.tar",
                  pass_context_configs=['tir.disable_vectorize=1']
                )

   


