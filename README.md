# HTVM
 _Efficient Neural Network Deployment on Heterogenous TinyML Platforms_

![CI](https://github.com/KULeuven-MICAS/htvm/actions/workflows/ci.yml/badge.svg)


HTVM is a deep learning compiler for deploying neural networks on heterogeneous embedded compute platforms with multiple scratchpad-managed accelerators.
HTVM generates self-contained C code that runs and dispatches neural network layers to either the platform's CPU, or one of it's accelerators.

To do this, HTVM mainly relies on:
* [Apache TVM](https://github.com/apache/tvm) to generate CPU kernels, and run different layers sequentially.
* [DORY](https://github.com/pulp-platform/dory) to generate accelerator kernels with optimized scratchpad memory management.

## Requirements

HTVM has many requirements, and due to the complex setup requirements we generally advise against installing locally.
It is recommended to use the container approach instead.
To use the container approach you need:
* Docker or Podman

HTVM mainly requires on the following:

* TVM (contained in this repository) and tools to compile TVM.
* [DORY](https://github.com/pulp-platform/dory) version `8a0fe7bcadb207c6d80820a4bd2c2f2c0e823248`
* Python 3.8

For DIANA, HTVM also requires:
* The adapted [PULP-SDK for DIANA](https://github.com/dianaKUL/pulp-sdk-diana)
* DORY Backend [kernels for DIANA](https://github.com/Aburrello/dory-hal)
* The [PULP RISC-V GNU Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/)

People who still wish to install locally, should take a look at this [Dockerfile](https://github.com/KULeuven-MICAS/htvm/blob/main/diana/docker/Dockerfile.tvm).

## Running HTVM inside of a container

This approach provides builds a container in which all requirements are installed and only requires you to build HTVM before you can get started. Here we use podman, but the instructions should also work with docker.


```sh
git clone --recursive https://github.com/KULeuven-MICAS/htvm
cd htvm
podman build . -f diana/docker/Dockerfile.tvm -t tvm-fork
```
After the container is created you can proceed with building the compiler (this only has to be done once):
```sh
podman run -itv=`pwd`:/tvm-fork:z tvm-fork
mkdir build
cp diana/config.cmake build
cd build
cmake ..
make -j$(nproc)
cd ..
```
Now you should be able to run the tests from inside the container
```sh
cd diana/byoc
pytest -v test.py
```
The HTVM compiler driver is now also available inside the container:
```sh
python3 /tvm-fork/diana/byoc/driver.py -h
```

## Project Status

HTVM currently supports deploying several neural networks on the [Diana heterogeneous SoC](https://doi.org/10.1109/ISSCC42614.2022.9731716).
A front-end with support for ingesting quantized neural networks from [Quantlib](https://github.com/pulp-platform/quantlib/) is work-in-progress.

## License

HTVM is Apache 2.0 Licensed.
## Acknowledgements

This repository started off as a fork of the [Apache TVM project](https://github.com/apache/tvm) on commit  `2af3ab1e36e0e78bac8448a0357abee317fabb1f` but was rebased on upstream several times.
