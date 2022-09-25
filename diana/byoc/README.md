BYOC for SIRIUS
===============

## Using the compiler

Code is already generated for the EWS operation to add a sirius device API for TVM using the [BYOC way](https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm).

This should work in this branch using when running the following:
```
python sirius/byoc/soma_codegen.py
```
Files are than generated in the /tmp folder. For EWS a function called `soma_add8()` is generated with should be implemented for the accelerator.o

## Extending the compiler

You can add operations as explained in the tutorial in:
`python/tvm/relay/op/contrib/soma.py`

And implement the visitor pattern in C++ in:
`src/relay/backend/contrib/soma/codegen.cc`




