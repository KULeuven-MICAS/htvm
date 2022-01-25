BYOC for SIRIUS
===============

Code is already generated for the EWS operation  to add a sirius device API for TVM using the [BYOC way](https://tvm.apache.org/2020/07/15/how-to-bring-your-own-codegen-to-tvm).

This should work in this branch using when running the following:
```
python sirius/byoc/soma_codegen.py
```
There seems to be a regression now which keeps it from working though :/

You can add operations as explained in the tutorial in:
`python/tvm/relay/op/contrib/soma.py`

And implement the visitor pattern in C++ in:
`src/relay/backend/contrib/soma/codegen.cc`




