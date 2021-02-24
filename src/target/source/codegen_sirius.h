/* Include once header guard */
#ifndef CODEGEN_SIRIUS_INCLUDED
#define CODEGEN_SIRIUS_INCLUDED

/********************************** METADATA **********************************/

/*
* Contributors: Vincent Tableau Roche
* Contacts: vincent.tableau@esat.kuleuven.be
* Creation Date: 2021-02-24
* Language: C++ Header
*/

/********************************** INCLUDES **********************************/

#include <tvm/target/codegen.h> // Used for various compilation types

#include "codegen_c.h"	// Used for inheritance

/****************************** CLASS DEFINITION ******************************/

namespace tvm {
namespace codegen {

// Our custom backend
class CodeGenSirius final : public CodeGenC {
 public:
  CodeGenSirius(); // Constructor.

  // Copied from codegen_c_host, used to gather the function names.
  void AddFunction(const PrimFunc& f);

  // Returns the name of the functions added to the module.
  Array<String> GetFunctionNames();

  // Sets up the compiler for the new module.
  void Init(bool output_ssa);

  // We are temporarily masking the code generation for assertion nodes since
  // it uses functions that won't be available on the Sirius platform.
  void VisitStmt_(const AssertStmtNode* op) final;

  // Overload the CallNode operation, required when compiling Resnet18 for error handling.
  void VisitExpr_(const CallNode* op, std::ostream& os) final;

  // Overloading the MinNode operation, as default backend uses imaginary min function.
  void VisitExpr_(const MinNode* op, std::ostream& os);
  
  // Overloading the MinNode operation, as default backend uses imaginary min function.
  void VisitExpr_(const MaxNode* op, std::ostream& os);
 
 protected:
  // The name of the module, printed at the beginning of the file.
  String module_name_;
  // Copied from CodeGenCHost, used to get the names of the functions in the module.
  Array<String> function_names_;
  // Some names which are supposed to live between the compilation of different modules.
  std::set<std::string> declared_globals_;
};
}  // namespace codegen
}  // namespace tvm

#endif	// end of include once header guard

/************************************ EOF *************************************/
