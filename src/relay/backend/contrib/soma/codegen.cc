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

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

/*! Extract arguments from the call node, and constuct the args vector. (Next functions.) */

std::vector<std::string> Conv2d(const CallNode* call) {
  std::vector<std::string> args;
  const auto* conv2d_attr = call->attrs.as<Conv2DAttrs>();
  CHECK(conv2d_attr);

  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());

  // Args: N, C, H, W
  for (auto s : ishape) {
    args.push_back(std::to_string(s));
  }

  // Args: Outputs, Groups, vertical padding, horizontal padding,
  // width, height, vertical stride, horizontal strid
  args.push_back(std::to_string(wshape[0]));
  args.push_back(std::to_string(conv2d_attr->groups)); // Used to parallelise the workload or to split operations.
  args.push_back(std::to_string(conv2d_attr->padding[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->padding[1].as<IntImmNode>()->value));
  args.push_back(std::to_string(wshape[2]));
  args.push_back(std::to_string(wshape[3]));
  args.push_back(std::to_string(conv2d_attr->strides[0].as<IntImmNode>()->value));
  args.push_back(std::to_string(conv2d_attr->strides[1].as<IntImmNode>()->value));

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
      // Get const: static_cast<float*>(dnnl_0_consts[0]->data)
      output.name = CreateDataReference(ext_func_id_, const_idx_);
      output.dtype = "float";

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

      const auto* type_node = cn->checked_type().as<TensorTypeNode>();
      CHECK(type_node);
      CHECK_EQ(GetDtypeString(type_node), "float") << "Only float is supported for now.";

      return {output};
    }

    std::vector<Output> VisitExpr_(const CallNode* call) final {
      GenerateBodyOutput ret;
      LOG(WARNING) << "Visiting call node with: " << AsText(call->op, false);
      if (const auto* func = call->op.as<FunctionNode>()) {
        ret = GenerateCompositeFunctionCall(func, call);
      } else {
        ret = GenerateOpCall(call);
      }

      buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
      ext_func_body_.push_back(ret.decl);
      return ret.outputs;
    }

    std::string JIT(const std::vector<Output>& out) {
      return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
    }
  private:
    struct GenerateBodyOutput {
      std::string decl;
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

    GenerateBodyOutput GenerateOpCall(const CallNode* call) {
      const auto* op_node = call->op.as<OpNode>();
      CHECK(op_node) << "Expect OpNode, but got " << call->op->GetTypeKey();
      using ArgFunType = std::function<std::vector<std::string>(const CallNode*)>;
      static const std::map<std::string, std::pair<std::string, ArgFunType>> op_map = {
          {"qnn.conv2d", {"soma_conv2d8", Conv2d}},
          {"qnn.dense", {"soma_dense8", Dense}},
          {"qnn.relu", {"soma_relu8", Relu}},
          {"qnn.batch_norm", {"soma_bn8", BatchNorm}},
          {"add", {"soma_add8", Add}},
      };

       const auto op_name = GetRef<Op>(op_node)->name;
       const auto iter = op_map.find(op_name);
       if (iter != op_map.end()) {
         return GenerateBody(call, iter->second.first, iter->second.second(call));
       }

       LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
       return {};
     }

     GenerateBodyOutput GenerateCompositeFunctionCall(const FunctionNode* callee,
                                                      const CallNode* caller) {
       const auto pattern_name = callee->GetAttr<runtime::String>(attr::kComposite);
       CHECK(pattern_name.defined()) << "Only functions with composite attribute supported";

       if (pattern_name == "soma.conv2d_bias_relu8") {
         const auto* conv_call =
             GetRootCall(callee->body.as<CallNode>(), 2, {"qnn.conv2d", "qnn.add", "qnn.relu"});
         return GenerateBody(conv_call, "soma_fused_conv2d_bias_relu8", GetArgumentNames(caller),
                             Conv2d(conv_call));
       } else if (pattern_name == "soma.conv2d_relu8") {
         const auto* conv_call = GetRootCall(callee->body.as<CallNode>(), 1, {"qnn.conv2d", "qnn.relu"});
         return GenerateBody(conv_call, "soma_fused_conv2d_relu8", GetArgumentNames(caller),
                             Conv2d(conv_call));
       }

       LOG(FATAL) << "Unknown composite function:" << pattern_name;
       return {};
     }

     GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                     const std::vector<std::string>& attribute_args) {
       return GenerateBody(root_call, func_name, GetArgumentNames(root_call), attribute_args);
     }

     GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                     const std::vector<std::string>& func_args,
                                     const std::vector<std::string>& attribute_args) {
       // Make function call with input buffers when visiting arguments
       CHECK_GT(func_args.size(), 0);
       std::ostringstream decl_stream;
       decl_stream << "(" << func_args[0];
       for (size_t i = 1; i < func_args.size(); ++i) {
         decl_stream << ", " << func_args[i];
       }

       // Analyze the output buffers
       std::vector<Type> out_types;
       if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
         auto type_node = root_call->checked_type().as<TupleTypeNode>();
         for (auto field : type_node->fields) {
           CHECK(field->IsInstance<TensorTypeNode>());
           out_types.push_back(field);
         }
       } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
         CHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
         out_types.push_back(root_call->checked_type());
       } else {
         LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
       }

       GenerateBodyOutput ret;
       for (const auto& out_type : out_types) {
         this->PrintIndents();
         const std::string out = "buf_" + std::to_string(buf_idx_++);
         const auto out_size = GetShape1DSize(out_type);
         decl_stream << ", " << out;

         Output output;
         output.name = out;
         output.size = out_size;
         output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
         output.need_copy = true;
         ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                               std::to_string(out_size) + ");");
         ret.outputs.push_back(output);
       }

       // Attach attribute arguments
       for (size_t i = 0; i < attribute_args.size(); ++i) {
         decl_stream << ", " << attribute_args[i];
       }
       decl_stream << ");";
       ret.decl = func_name + decl_stream.str();
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
      code_stream_ << "\/\/ Generated by the TVM SOMA codegen.\n";

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
      const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
      CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";

      std::cout << "HEllo1" << std::endl;
      auto ret = (*pf)(code, "c", syms, variables);
      std::cout << "Hello2" << std::endl;
      return ret;
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

