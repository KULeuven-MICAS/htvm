!rm golden_model_output.txt
set print elements 0
set print repeats 0
set pagination off
file build/demo
break demo.c:36
run
set logging file golden_model_output.txt
print /d *output@output_size
set logging off
quit
