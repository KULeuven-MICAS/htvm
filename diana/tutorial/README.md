Diana TVM-fork tutorials
========================

This folder contains some tutorials which are supposed to be run inside a jupyterlab notebook server.

## How to use

Before you start:
1. Clone this repository.
2. Then download the artefacts of a recent succesful build of the `build_tvm_release` job in the CI.
3. Extract the contents of the artefacts in `/tvm-fork/build`.

The server can be started with the following command executed in the root of this repository, (assuming you have an image of `tvm-fork-tuto` on your system):
```bash
sudo docker -itp 8888:8888 -v=`pwd`:/tvm-fork:z tvm-fork-tuto
```
Note that on rootless setups the `sudo` command is optional.

Once you have launched this jupyterlab server, go to the link it outputs in the terminal, and open it in your browser:
[http://127.0.0.1:8888/lab?token=this-token-changes-for-every-run](http://127.0.0.1:8888/lab)

## Other options, notes

* if you want to run multiple containers with different jupyterlab servers, you can change the first argument in the port specifier (the `-p` part of your docker command, change for example to `8889:8888`).
In that case you do have to alter the link though, (e.g. `http://127.0.0.1:8888` becomes `http://127.0.0.1:8889`)

* You can also use the container to run the experiments without the notebook server:
```bash
sudo docker -it -v `pwd`:/tvm-fork:z tvm-fork-tuto bash
```

* You can also attach other repositories to the `/tvm-fork` mounting point iside the container, just change the first part of your `-v` string. (e.g. `-v=abs_path_to_my_other_repo:/tvm-fork:z`).
You can also just use the above command (with `pwd`) in the root of a different directory.
