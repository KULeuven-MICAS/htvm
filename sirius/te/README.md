# TVM Tensor Expressions (TE) API examples

This folder contains some examples for using the TVM TensorExpressions (TE) API.
TE is a low level API for the TVM stack. 
It is available from C++ and Python. We currently only use the Python API.

## Contents

* Examples for tensorization:
    * `ews_te.py` element-wise sum tensorized with a 2x2 matrix intrinsic using `tile`.
    * `ews_te_assign.py` variation of element-wise sum tensorization with a custom intrinsic.
    * `ews_te_soma.py` element-wise sum tensorized with a 4xHxC tensor intrinsic using `split`.

---

## Introduction

TensorExpressions is based on [Halide](https://halide-lang.org/) and uses its decoupled compute/schedule paradigm.

TE is relatively well documented in [TVM's documentation](https://tvm.apache.org/docs/tutorials/index.html#tensor-expression-and-schedules). Some useful resources:
* [TE get started tutorial](https://tvm.apache.org/docs/tutorials/get_started/tensor_expr_get_started.html#sphx-glr-tutorials-get-started-tensor-expr-get-started-py)
* [List of all schedule primitives in TE](https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html#sphx-glr-tutorials-language-schedule-primitives-py)
* [TE Tensorization tutorial](https://tvm.apache.org/docs/tutorials/language/tensorize.html#sphx-glr-tutorials-language-tensorize-py)
* [Relay Op strategy tutorial](https://tvm.apache.org/docs/dev/relay_op_strategy.html): how to write your own Relay Strategy
* [Tuple input schedule tutorial](https://tvm.apache.org/docs/tutorials/language/tuple_inputs.html#sphx-glr-tutorials-language-tuple-inputs-py) (often used in Relay Strategies)

An introduction to TE can be found in the [TVM documentation](https://tvm.apache.org/docs/tutorials/get_started/tensor_expr_get_started.html).


The Relay lowering process follows the relay strategy which defines what schedules for what operators to use in what case.
e.g. If you call an element-wise sum in relay, it should use the optimized element-wise sum operator for sirius
(the one that calls the element-wise sum in the HW library). In this context we use TE for:

* the definition of the Relay strategy.
  This basically tells TVM how to lower relay operator nodes to Target machine code. 
  Such a strategy is thus target dependent. 
  The Relay strategy for SIRIUS is described in `tvm/python/relay/op/strategy/sirius.py` and contains a lot of TE code.
  
* The optimized versions of operators. Often we use a tensorization primitive on this schedule.
  Tensorization replaces parts of the schedule with a certain intrinsic (e.g. a function call to the HW library).
  Optimized operators are thus also Target dependent. SIRIUS optimized operators are described in `tvm/python/topi/sirius/`
  
Besides the strategies and operators we created for SIRIUS it is instructive to look into operators for other targets defined in:
`tvm/python/topi/` and strategies for other targets in `tvm/python/relay/op/strategy`.

**Please note**:
**Working on operators in topi and the sirius strategy requires rerunning the python install command in the 
`tvm/python/setup.py` for the changes to reflect in the tvm package that python uses.**

---

## Walkthrough for `ews_te.py`