# HTVM
 _Efficient Neural Network Deployment on Heterogenous TinyML Platforms_

[![CI](https://github.com/KULeuven-MICAS/htvm/actions/workflows/ci.yml/badge.svg)](https://github.com/KULeuven-MICAS/htvm/actions)


HTVM is a deep learning compiler for deploying neural networks on heterogeneous embedded compute platforms with multiple scratchpad-managed accelerators.
HTVM generates self-contained C code that runs and dispatches neural network layers to either the platform's CPU, or one of it's accelerators.

To do this, HTVM mainly relies on:
* [Apache TVM](https://github.com/apache/tvm) to generate CPU kernels, and run different layers sequentially.
* [DORY](https://github.com/pulp-platform/dory) to generate accelerator kernels with optimized scratchpad memory management.

## Requirements

Main requirements:

* TVM (contained in this repository) and tools to compile TVM.
* [DORY](https://github.com/pulp-platform/dory) version `8a0fe7bcadb207c6d80820a4bd2c2f2c0e823248`
* Python 3.8

For DIANA, HTVM also requires:
* The adapted [PULP-SDK for DIANA](https://github.com/dianaKUL/pulp-sdk-diana)
* DORY Backend [kernels for DIANA](https://github.com/Aburrello/dory-hal)
* The [PULP RISC-V GNU Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/)

For your convenience, we advise to use our docker container with all dependencies installed, needed for building TVM.

## Installation in a container

We use `podman` commands here, but note that you can use `docker` as well if preferred.

### Getting the docker image

Our github CI has an up-to-date image available that you can pull with:
```sh
podman pull ghcr.io/kuleuven-micas/htvm:main
```

Or you could build the container image locally with:
```sh
git clone --recursive https://github.com/KULeuven-MICAS/htvm
cd htvm
podman build . -f diana/docker/Dockerfile.tvm -t htvm:main
```

> [!NOTE]
> See the [Dockerfile](https://github.com/KULeuven-MICAS/htvm/blob/main/diana/docker/Dockerfile.tvm) in case you want to attempt installation without a container.

### Building HTVM

If you haven't already cloned the repo, do:
```sh
git clone --recursive https://github.com/KULeuven-MICAS/htvm
cd htvm
```

Now create and start a container:
```sh
podman run -itv=`pwd`:/tvm-fork:z htvm:main
```

Inside the container shell run:

```sh
mkdir build
cp diana/config.cmake build
cd build
cmake ..
make -j$(nproc)
cd ..
```

Test if it works (also run from inside the container):

```sh
cd diana/byoc
python3 driver.py -h
```

## Compiling an ONNX model

A number of ONNX example models, quantized by [diana-quantlib](https://github.com/KULeuven-MICAS/diana-quantlib), are provided in this repo through git LFS.
For quantizing your own models, see [diana-quantlib](https://github.com/KULeuven-MICAS/diana-quantlib).

Download the model data with:

```sh
git lfs pull
```

Compile a model for DIANA with digital acceleration:
```sh
python3 driver.py --no-run --onnx test_data/export_resnet8/ResNet_QL_NOANNOTATION.onnx
```

Output C-code and pulp binaries can be found at `/tmp/digital_pulp_dory_fused_O3_None/pulp/`.

Compiling a model for running on the CPU of your local machine:
```sh
python3 driver.py --no-run --device x86 --target c --onnx test_data/export_resnet8/ResNet_QL_NOANNOTATION.onnx
```

Output C-code and x86 binaries can be found at `/tmp/digital_x86_c_fused_O3_None/x86`.

Run it locally with:
```sh
/tmp/digital_x86_c_fused_O3_None/x86/demo
```

## Running tests

In addition to the standard test suite, provided by TVM, HTVM contains its own additional unit tests and end-to-end test.

The unit tests can be run with:
```sh
cd /path/to/htvm
pytest -v tests/python/contrib/test_soma_dory
```

The end-to-end tests rely on example ONNX files that are tracked with git lfs. Run `git lfs pull` in case you haven't done that already.
Now run:
```sh
cd diana/byoc
pytest -v test.py
```

## Project Status

HTVM currently supports deploying a number of tested neural networks on the [Diana heterogeneous SoC](https://doi.org/10.1109/ISSCC42614.2022.9731716).

The front-end supports ingesting quantized neural networks in ONNX format from [Quantlib](https://github.com/pulp-platform/quantlib/).

## Publications

[HTVM was presented at the 60th Design Automation Conference (DAC)](https://doi.org/10.1109/DAC56929.2023.10247664).
If you use this work in your publication, please cite:
```bibtex
@INPROCEEDINGS{10247664,
  author={Van Delm, Josse and Vandersteegen, Maarten and Burrello, Alessio and Sarda, Giuseppe Maria and Conti, Francesco and Pagliari, Daniele Jahier and Benini, Luca and Verhelst, Marian},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)}, 
  title={HTVM: Efficient Neural Network Deployment On Heterogeneous TinyML Platforms}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/DAC56929.2023.10247664}}
```

## License

HTVM is Apache 2.0 Licensed.
## Acknowledgements

This repository started off as a fork of the [Apache TVM project](https://github.com/apache/tvm) on commit  `2af3ab1e36e0e78bac8448a0357abee317fabb1f` but was rebased on upstream several times.
