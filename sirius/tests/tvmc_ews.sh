python3 -m tvm.driver.tvmc compile --target="soma, c" \
            --runtime=crt \
            --executor=aot \
            --executor-aot-interface-api=c \
            --executor-aot-unpacked-api=1 \
            --pass-config tir.disable_vectorize=1 \
            sirius/byoc/ews_net.onnx -f mlf
