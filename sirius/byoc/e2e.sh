#!/usr/bin/env bash

CALLDIR=$(pwd)
BASEDIR=$(dirname "$0")
echo $BASEDIR

# Go to byoc dir
cd $BASEDIR

# Make clean
rm -rf build
rm -rf tfmodel
rm model.tflite

mkdir -p build

echo "Create TF model"

python ews_net_tf.py

echo "Perform TVM compilation"

python -m tvm.driver.tvmc compile --target="soma, c" \
	     --runtime=crt \
             --executor=aot \
             --executor-aot-interface-api=c \
             --executor-aot-unpacked-api=1 \
             --pass-config tir.disable_vectorize=1 \
             ./model.tflite -f mlf -o build/model.tar

cd build

tar -xf model.tar

# Go back to caller directory
cd $CALLDIR
