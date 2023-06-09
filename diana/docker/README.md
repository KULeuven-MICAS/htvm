# Dockerfiles

This folder contains two Dockerfiles:
* `Dockerfile.tvm`: Dockerfile that installs all dependencies for compiling and running `tvm-fork`.
**Note:** this container image does *NOT* include the `tvm-fork` repository itself.
* `Dockerfile.tutorial`: Dockerfile that installs extra dependencies on top of `Dockerfile.tvm` for running the tutorials in `/tvm-fork/diana/tutorial/`.
It installs `jupyterlab` for running jupyter notebooks in the web browser.
**Note:** this container image does *NOT* include the `tvm-fork` repository itself

It should be possible to use both of these images with `docker` and `podman`.
We mainly use rootless `podman` inside of this project and also in this README.
**Note:** If you want to use docker instead, you'll probably have to use `sudo docker` instead of `podman`.

# Building and using the docker images

## 1. `Dockerfile.tvm`

This image is also used to test our fork's inner workings and is built and pushed in CI.
If you have access to the [container registry of this Gitlab Repository](https://gitlab.com/soma_compiler/tvm-fork/container_registry),
you might want to pull this image instead of building it yourself.

In the root directory of this repository (`tvm-fork`) execute the following command:
```sh
$ podman build . -f diana/docker/Dockerfile.tvm -t tvm-fork
```
Now podman should build the image for you.
Afterwards you can run the container by executing this command in the root of the repo (`tvm-fork`):
```sh
$ podman run -it -v=`pwd`:/tvm-fork:z tvm-fork
```
This will mount the folder containing the repository on your host file system inside the container's file system at `/tvm-fork`.
You can edit the files inside or outside of the container and the changes will propagate to both environments. 

To exit the container, just type `exit` inside of the terminal.


## 2. `Dockerfile.tutorial`

To build this you need an image of `tvm-fork` (built with `Dockerfile.tvm`).
```
Then you can build with:
```
$ podman build . -f diana/docker/Dockerfile.tutorial -t tvm-fork-tuto

```

This container image can be used in a few different ways:
1. `terminal`: from the root of the repository run:
	```sh
	$ podman run -it -v=`pwd`:/tvm-fork:z tvm-fork-tuto bash
	```
	This exposes your host repository files to the container (like mentioned above).
2. `jupyterlab`: from the root of the repository run:
	```sh
	$ podman run -ip 8888:8888 -v=`pwd`:/tvm-fork:z tvm-fork-tuto
	```
	This exposes your host repository files to the container (like mentioned above) and also
	exposes port 8888 of the container to port 8888 of the host.
	You should now be able to access the `jupyterlab` interface by clicking the link that starts with `https://127.0.0.1:8888/...`
3. If you want to use gdb from inside the container, you can bind the local domain to the container's and connect to a debugging session over TCP like so:
    ```sh
    $ podman run -it --net=host -v=`pwd`:/tvm-fork:z tvm-fork-tuto bash
    ``` 
    Note that using the `--net=host` imposes certain security risks which should be taken into account.

**Note: Be extra careful when using SSH credentials inside of the container with jupyterlab.
If you expose jupyterlab without a password or token (turned on by default),
people with access to jupyterlab can access an SSH terminal on your behalf.
If you are unsure about the security risks, please do not expose SSH credentials 
inside of the container, but use a separate terminal on the host instead.**

