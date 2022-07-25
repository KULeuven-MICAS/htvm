//
// Created by wvr on 26.10.21.
// Based on: https://github.com/apache/tvm/blob/8a0249cd4d12a2eb1a4e7a692a9265bc63fec5c8/src/relay/backend/contrib/dnnl/codegen.cc  */
//
//

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "../../../transforms/pattern_utils.h"

#include "../codegen_c/codegen_c.h"
#include "../../../../target/source/codegen_c.h"
#include "../../../../target/source/codegen_c_host.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

/*! Extract arguments needed to make a soma call and put them in a map */

std::map<std::string, int> GetConv2dAttributes(const FunctionNode* func_call) {
    std::map<std::string, int> args;

    const auto* final_call = func_call->body.as<CallNode>();
    const auto* conv_call = GetRootCall(func_call->body.as<CallNode>(), "qnn.conv2d");
    const auto* shift_call = GetRootCall(func_call->body.as<CallNode>(), "right_shift");

    const auto* conv2d_attr = conv_call->attrs.as<Conv2DAttrs>();
    CHECK(conv2d_attr);

    auto ishape = GetShape(conv_call->args[0]->checked_type());
    auto wshape = GetShape(conv_call->args[1]->checked_type());
    auto oshape = GetShape(shift_call->args[0]->checked_type());  // output shape of conv2d is input shape of shift op

    // get conv_params (assume the ranges are already checked)
    args.insert({"pad_up_down", conv2d_attr->padding[0].as<IntImmNode>()->value});
    args.insert({"pad_left_right", conv2d_attr->padding[1].as<IntImmNode>()->value});

    args.insert({"conv_strided", 0});
    if (conv2d_attr->strides[0].as<IntImmNode>()->value > 1) {
        args["conv_strided"] = 1;   // only support symmetric strides 1 or 2
    }

    if (IsOp(final_call, "clip")) {
        args.insert({"activation_function", 1});
    }

    int right_shift =  GetScalarFromConstant<int>(shift_call->args[1]);
    args.insert({"shift_fixed_point", right_shift});

    // get input_dims
    args.insert({"input_dims_c", ishape[1]});
    args.insert({"input_dims_h", ishape[2]});
    args.insert({"input_dims_w", ishape[3]});

    // get filter_dims: TODO: layout conversion may be needed
    args.insert({"filter_dims_k",  wshape[0]});
    args.insert({"filter_dims_c",  wshape[1]});
    args.insert({"filter_dims_fy", wshape[2]});
    args.insert({"filter_dims_fx", wshape[3]});

    // get output_dims
    args.insert({"output_dims_c", oshape[1]});
    args.insert({"output_dims_h", oshape[2]});
    args.insert({"output_dims_w", oshape[3]});

    return args;
}

std::vector<std::string> Dense(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: Input shape dimensions (N, C), weight/output shape O
  args.push_back(std::to_string(ishape[0]));
  args.push_back(std::to_string(ishape[1]));
  args.push_back(std::to_string(wshape[0]));

  return args;
}

std::vector<std::string> Relu(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W (all tensor dimensions)
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  return args;
}

std::vector<std::string> BatchNorm(const CallNode* call) {
  std::vector<std::string> args;
  const auto* bn_attr = call->attrs.as<BatchNormAttrs>();
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: N, C, H, W (all tensor dimensions)
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: epsilon
  args.push_back(std::to_string(bn_attr->epsilon));

  return args;
}

std::vector<std::string> Add(const CallNode* call) {
  std::vector<std::string> args;
  auto ishape = GetShape(call->args[0]->checked_type());

  // Args: H, W (input size = output size)
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }
  // Hardcode 8 as precision
  // TODO make this not hardcoded
  args.push_back("8");

  return args;
}


/*!
 * Generate code for the Relay functions that are offloaded to the SOMA codegen.
 */
class CodeGenSOMA : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {
  public:
    explicit CodeGenSOMA(const std::string& id) { this->ext_func_id_ = id; }

    std::vector<Output> VisitExprDefault_(const Object* op) final {
      LOG(FATAL) << "SOMA codegen doesn't support: " << op->GetTypeKey();
      return {};
    }

    std::vector<Output> VisitExpr_(const VarNode* node) final {
      ext_func_args_.push_back(GetRef<Var>(node));
      Output output;
      output.name = node->name_hint();
      return {output};
    }

    std::vector<Output> VisitExpr_(const TupleNode* node) final {
      std::vector<Output> outs;
      for (auto field : node->fields) {
        auto res = VisitExpr(field);
        CHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
        outs.push_back(res[0]);
      }
      return outs;
    }

    std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
      auto res = VisitExpr(op->tuple);
      CHECK_GT(res.size(), static_cast<size_t>(op->index));

      // Only keep the item we want for the child node.
      // FIXME(@comaniac): The other items should still be requried for the primary outputs.
      return {res[op->index]};
    }

    std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
      Output output;

      const auto* type_node = cn->checked_type().as<TensorTypeNode>();
      //CHECK(type_node);
      //CHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

      output.dtype = GetDtypeString(type_node);

      output.name = CreateDataReference(ext_func_id_, const_idx_);

      // Generate the global variable for needed ndarrays
      if (const_array_name_.empty()) {
        const_array_name_ = CreateNDArrayPool(ext_func_id_);
        std::string checker = CreateInitChecker(ext_func_id_);
        ext_func_body_.insert(ext_func_body_.begin(), checker);
      }

      // Give the ndarray a unique name to ease the initialization of it at
      // runtime.
      std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
      const_vars_.push_back(const_var_name);
      const_idx_++;
      return {output};
    }

    std::vector<Output> VisitExpr_(const CallNode* call) final {
      GenerateBodyOutput ret;
      LOG(WARNING) << "Visiting call node with: " << AsText(call->op, false);
      if (const auto* func = call->op.as<FunctionNode>()) {
        ret = GenerateCompositeFunctionCall(func, call);
      } else {
        // TODO: raise an error since we don't accelerate non partitioned calls
      }

      buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
      ext_func_body_.insert(ext_func_body_.end(), ret.decl.begin(), ret.decl.end());    // extend this vector
      return ret.outputs;
    }

    std::string JIT(const std::vector<Output>& out) {
      return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
    }

  protected:
    /*!
    * \brief A common interface that is used by various external runtime to
    * generate the wrapper to invoke external kernels.
    *
    * \param ext_func_id The unique id of an external function. It will be used
    * during runtime to pick the correct external function.
    * \param args The arguments used by the external function.
    * \param buf_decl The declaration of temporary buffers that used to store the
    * intermeidate of each external kernel.
    * \param body The statements of the external function.
    * \param out The name and id pairs for output.
    *
    * \return The emitted code string.
    */

    /*based off of codegen_c/codegen_c.h*/
    std::string JitImpl(const std::string& ext_func_id, const Array<Var>& args,
          	      const std::vector<std::string>& buf_decl,
          	      const std::vector<std::string>& body, const std::string& const_arr_name,
          	      const std::vector<Output>& outs) {
      // Create a declaration for global ndarrays that contain constant data.
      if (!const_arr_name.empty()) {
        code_stream_ << "#ifdef __cplusplus\n";
        code_stream_ << const_arr_name << "\n\n";
        code_stream_ << "#endif\n";
      }
      // Create the signature. For example, it could be:
      // void dnnl_0_(float* in0, float* in1, float* out0, float* out1) {}
      // TODO I removed this "_". Is this okay?
      //code_stream_ << "void " << ext_func_id << "_(";
      code_stream_ << "int32_t " << ext_func_id << "(";

      for (const auto& arg : args) {
        const auto& dtype_str = GetDtypeString(arg);
        code_stream_ << dtype_str << "* " << arg->name_hint() << ", ";
      }
      for (size_t i = 0; i < outs.size() - 1; ++i) {
        code_stream_ << outs[i].dtype << "* out" << i << ", ";
      }
      code_stream_ << outs.back().dtype << "* out" << outs.size() - 1 << ") {\n";
      this->EnterScope();

      // Function body
      for (auto decl : buf_decl) {
        this->PrintIndents();
        code_stream_ << decl << "\n";
      }
      code_stream_ << "\n";
      for (auto stmt : body) {
        this->PrintIndents();
        code_stream_ << stmt << "\n";
      }

      // Copy output
      for (size_t i = 0; i < outs.size(); ++i) {
        if (!outs[i].need_copy) {
          continue;
        }
        this->PrintIndents();
        code_stream_ << "memcpy(out" << i << ", " << outs[i].name << ", " << outs[i].size
          	         << " * sizeof(" << outs[i].dtype << "));\n";
      }

        // Free buffers
        for (size_t i = 0; i < buf_decl.size(); i++) {
          this->PrintIndents();
          code_stream_ << "free(buf_" << i << ");\n";
        }

        this->ExitScope();
        code_stream_ << "}\n";

        // Create the wrapper to call the ext_func
        // TODO I just disabled this :D
        //this->GenerateBackendCFunc(ext_func_id, args, const_arr_name, outs);
        return code_stream_.str();
      }

    std::string GetDtypeString(const Var& var) {
        auto ttype = var->checked_type().as<TensorTypeNode>();
        ICHECK(ttype) << "Expect TensorTypeNode";
        return GetDtypeString(ttype);
      }

  private:
    struct GenerateBodyOutput {
      std::vector<std::string> decl;
      std::vector<std::string> buffers;
      std::vector<Output> outputs;
      };

      std::vector<std::string> GetArgumentNames(const CallNode* call) {
        std::vector<std::string> arg_names;
        for (size_t i = 0; i < call->args.size(); ++i) {
          auto res = VisitExpr(call->args[i]);
          for (const auto& out : res) {
            arg_names.push_back(out.name);
          }
        }
        return arg_names;
      }

    /*!
    * \brief Returns dtype string, extended drom CodeGenCBase to also support int8
    *
    * \param ttype TensorTypeNode* to get the dtype of
    *
    * \return The dtype string.
     */
    std::string GetDtypeString(const TensorTypeNode* ttype) {
      std::string dtype;
      LOG(WARNING) << "GetDtypeString dtype";
      if (runtime::TypeMatch(ttype->dtype, kDLInt, 8)) {
        LOG(WARNING) << "GetDtypeString dtype = int8";
        dtype = "int8_t";
      } else {
        dtype = CodegenCBase::GetDtypeString(ttype);
      }
      return dtype;
    }

    GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                     const CallNode* caller) {
        const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
        CHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

        if (pattern_name == "soma.qnn_conv2d") {
            auto attribute_args = GetConv2dAttributes(callee);
            return GenerateConv2dBody(callee->body.as<CallNode>(), GetArgumentNames(caller), attribute_args);
        }

        LOG(FATAL) << "Unknown composite function:" << pattern_name;
        return {};
    }

    GenerateBodyOutput GenerateConv2dBody(const CallNode* final_call,
                                          const std::vector<std::string>& func_args,
                                          std::map<std::string, int>& attribute_args) {
       /*
        * Example output:
        *
        * const SomaConvParams conv_params = {.activation_function = 1, shift_fixed_point = 19, .pad_up_down = 1, .pad_left_right = 1, .conv_strided = 0};
        * const SomaDataShape input_dims = {.c = 3, .h = 10, .w = 20};
        * const SomaFilterShape filter_dims = {.c = 3, fy = 3, fx = 3, k = 5};
        * const SomaFilterShape output_dims = {.c = 5, .h = 10, .w = 20};
        * return soma_conv2d_s8(&conv_params, &input_dims, input_data, &filter_dims, filter_data, 5, bias_data, &output_dims, output_data);
        */
        GenerateBodyOutput ret;
        std::ostringstream decl_stream;

        // define conv_params
        decl_stream << "const SomaConvParams conv_params = {" \
                    << ".activation_function = " << attribute_args["activation_function"] << ", " \
                    << ".shift_fixed_point = " << attribute_args["shift_fixed_point"] << ", " \
                    << ".pad_up_down = " << attribute_args["pad_up_down"] << ", " \
                    << ".pad_left_right = " << attribute_args["pad_left_right"] << ", " \
                    << ".conv_strided = " << attribute_args["conv_strided"] << "};";
        ret.decl.push_back(decl_stream.str());
        decl_stream.str(std::string());

        // define input_dims
        decl_stream << "const SomaDataShape input_dims = {" \
                    << ".c = " << attribute_args["input_dims_c"] << ", " \
                    << ".h = " << attribute_args["input_dims_h"] << ", " \
                    << ".w = " << attribute_args["input_dims_w"] << "};";
        ret.decl.push_back(decl_stream.str());
        decl_stream.str(std::string());

        // define filter_dims
        decl_stream << "const SomaFilterShape filter_dims = {" \
                    << ".c = " << attribute_args["filter_dims_c"] << ", " \
                    << ".fy = " << attribute_args["filter_dims_fy"] << ", " \
                    << ".fx = " << attribute_args["filter_dims_fx"] << ", " \
                    << ".k = " << attribute_args["filter_dims_k"] << "};";
        ret.decl.push_back(decl_stream.str());
        decl_stream.str(std::string());

        // define output_dims
        decl_stream << "const SomaDataShape output_dims = {" \
                    << ".c = " << attribute_args["output_dims_c"] << ", " \
                    << ".h = " << attribute_args["output_dims_h"] << ", " \
                    << ".w = " << attribute_args["output_dims_w"] << "};";
        ret.decl.push_back(decl_stream.str());
        decl_stream.str(std::string());

        // make function call
        decl_stream << "return soma_conv2d_s8(&conv_params, &input_dims, " \
                    << func_args[0] << ", &filter_dims, " \
                    << func_args[1] << ", " \
                    << attribute_args["filter_dims_k"] << ", " \
                    << func_args[2] << ", &output_dims, " << "out0" << ");";
        ret.decl.push_back(decl_stream.str());
        decl_stream.str(std::string());

        // set output buffer type
        auto out_type = final_call->checked_type();
        Output output;
        output.name = "out0";
        output.size = GetShape1DSize(out_type);
        output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
        output.need_copy = false;
        ret.outputs.push_back(output);

        return ret;
    }

    /*! \brief The id of the external soma ext_func. */
    std::string ext_func_id_{""};

    /*!
     * \brief The index to track the output buffer. Each kernel will redirect the
     * output to a buffer that may be consumed by other kernels.
     */
    int buf_idx_{0};
    /*! \brief The index of global constants. */
    int const_idx_{0};
    /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
    Array<Var> ext_func_args_;
    /*! \brief Statement of the function that will be compiled using DNNL kernels. */
    std::vector<std::string> ext_func_body_;
    /*! \brief The array declared to store the constant values. */
    std::string const_array_name_;
    /*! \brief The declaration of intermeidate buffers. */
    std::vector<std::string> buf_decl_;
    /*! \brief The variable name to constant mapping. */
    Array<String> const_vars_;

    friend class SOMAModuleCodegen;
};


class SOMAModuleCodegen : public CSourceModuleCodegenBase {
  public:
    /*!
     * Create a corresponding SOMA function for the given relay Function.
     * @param func The function to generate code for.
     * @return Tuple of constant weights (WVR: I don't know what that means)
     * (symbols, values).
     */
    std::pair<std::string, Array<String>> GenSOMAFunc(const Function& func) {
      CHECK(func.defined()) << "Input error: expect a Relay function.";

      // Record the external symbol for runtime lookup.
      auto sid = GetExtSymbol(func);

      CodeGenSOMA builder(sid);
      auto out = builder.VisitExpr(func->body);
      code_stream_ << builder.JIT(out);

      return {sid, builder.const_vars_};
    }

    /*!
     * \brief The overridden function that will create a CSourceModule. In order
     * to compile the generated C source code, users need to specify the paths to
     * some libraries, including some TVM required and dnnl specific ones.
     *
     * \param ref An object ref that could be either a Relay function or module.
     *
     * \return The runtime module that contains C source code.
     */
    runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
      code_stream_ << "// Generated by the TVM SOMA codegen.\n";
      code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
      code_stream_ << "#include <soma_wrapper.h>\n\n";

      // "ref" should be the paritioned Relay function with kCompiler=dnnl.
      CHECK(ref->IsInstance<FunctionNode>());
      auto res = GenSOMAFunc(Downcast<Function>(ref));

      // "code" is the generated C code with SOMA APIs.
      std::string code = code_stream_.str();

      // "res" is a tuple of constant weights (symbols, values).
      // All constant tensors will be serialzied along with the generated C code
      // when export_library is invoked.
      String sym = std::get<0>(res);
      Array<String> variables = std::get<1>(res);
      Array<String> syms = {sym};

      // Create a CSource module with all above artifacts.
      //const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
      //CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
      //auto ret = (*pf)(code, "c", syms, variables);
      //return ret;
      return codegen::CSourceModuleCreate(code, "c", syms);
    }

  private:
    /*!
     * \brief The code stream that prints the code that will be compiled using
     * external codegen tools.
     */
    std::ostringstream code_stream_;
};


/*!
 * \brief Entry function for the SOMA compiler...
 *
 * \param ref An object ref that could be either a Relay function or module.
 *
 * \return The runtime module that contains C source code.
 */
runtime::Module SOMACompiler(const ObjectRef& ref) {
  SOMAModuleCodegen soma;
  return soma.CreateCSourceModule(ref);
}


TVM_REGISTER_GLOBAL("relay.ext.soma").set_body_typed(SOMACompiler);


}
}
}

