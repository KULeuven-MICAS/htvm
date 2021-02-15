# TVM fork for Sirius

This folder contains all files related to Sirius development

## Installation
Specific installation instructions are put here.
It closely resembles https://tvm.apache.org/docs/install/from_source.html yet with some recommendations and other Sirius-specifics.
We are assuming you are installing on a linux machine.

---

# Install from Source

This page gives instructions on how to build and install the TVM package from
scratch on various systems. It consists of two steps:

1. First build the shared library from the C++ codes (`libtvm.so` for linux).
2. Setup for the language packages (e.g. Python Package).

## Developers: Get Source from Gitlab

Clone the git repository from gitlab.
It is important to clone the submodules along, with ``--recursive`` option.
Clone with ssh (recommended):
```bash
    git clone --recursive git@gitlab.com:JosseVanDelm/tvm-fork.git
```
Or clone with https:
```bash
    git clone --recursive https://gitlab.com/JosseVanDelm/tvm-fork.git
```


## Build the Shared Library

Our goal is to build the shared libraries on Linux (the target library are `libtvm.so`)
These instructions are specific to Ubuntu or Ubuntu-derived distributions.
```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```
The minimal building requirements are

- A recent c++ compiler supporting C++ 14 (g++-5 or higher)
- CMake 3.5 or higher
- We highly recommend to build with LLVM to enable all the features.
- ~~If you want to use CUDA, CUDA toolkit version >= 8.0 is required. If you are upgrading from an older version, make sure you purge the older version and reboot after installation.~~ We don't need CUDA.

We use cmake to build the library.
The configuration of TVM can be modified by `config.cmake`.

- First create a build directory, copy the ``cmake/config.cmake`` to the directory.
A Sirius-specific one is provided in this folder, .
  ```bash
      mkdir build
      cp sirius/config.cmake build
  ```

- We can then build tvm and related libraries. `-j4` should be adapted to the available threads on your system. e.g. if you have 24 threads, you can use `-j24` for faster compilation.
  ```bash
      cd build
      cmake ..
      make -j4
  ```
If everything goes well, we can go to :ref:`python-package-installation`

## Python package installation

In this step we will install the TVM python package

It is recommended to install your packages in an isolated development environment for this project.
We will be using `pyenv-virtualenv` to create such an environment, however other tools may be used to achieve the same purpose (like `conda` or `pipenv`).

`pyenv` is useful because it can easily switch python versions to make sure we're all using the same version. In this tutorial we will be using `python 3.7.7`

You can create a new virtualenv by using this command:
```bash
pyenv virtualenv  3.7.7 tvm-sirius
```
Now we can activate the tvm-sirius virtualenv for isolated python development.
This environment has to be activated every time you open a new terminal.
```bash
pyenv activate tvm-sirius
```
The python package is located at `tvm/python`
We can now install the python package by using:
Install TVM python bindings by `setup.py`:
```bash
cd python; python setup.py install; cd ..
```
Python dependencies that also need to be installed
```bash
pip3 install numpy decorator attrs
```
