# Dockerfiles

**These methods are now deprecated, for proper usage of Docker containers check .gitlab-ci.yml**
_Note: The dockerfile.pulp builds the RISC-V toolchain, while a prebuilt version is available from https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/tag/v1.0.16_

This folder contains two Dockerfiles:
* `Dockerfile.pulp`: for running the pulp risc-v gcc c compiler.
* ~`Dockerfile.tvm`: for running this repository.~

## Usage
- Create an empty directory `mkdir directory`.
- Copy the `Dockerfile.<x>` in the new directory and rename to `Dockerfile`.
- Copy your public and private ssh keys to in the directory. They are needed to perform the `git clone --recursive`. 
  Note that you can use a dedicated set of keys for the docker containers and add them to your GitHub and GitLab 
  accounts. The keys should be called `docker_ed25519` and `docker_ed25519.pub` and should use the right algorithm.
- run `docker build -t <tag> .` , with `<tag>` being the name you want to give to the docker image.

- To start the docker in interactive mode (like a VM), use `docker run -it --name=<container> <tag> bash` with `<tag>` 
  being the name of the image mentioned before and `<container>` being the name given to the container.

- Next time you want to reuse the same container, just use `docker start -i <container>`.
