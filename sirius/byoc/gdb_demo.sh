!rm demo.txt
set print elements 0
set print repeats 0
set pagination off
file build/pulpissimo/demo/demo
target remote localhost:3333
load
break tvmgen_default_run
c
n
n
set logging file demo.txt
set logging on
print /d *output@output_size
set logging off
