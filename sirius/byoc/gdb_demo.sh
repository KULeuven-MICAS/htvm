!rm gdb.txt
set print elements 0
set print repeats 0
set pagination off
file build/pulpissimo/demo/demo
target remote localhost:3333
load
break demo.c:36
c
file build/demo
set logging file golden_model_output.txt
print /d *output@output_size
set logging off
