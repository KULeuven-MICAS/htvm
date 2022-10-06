!rm demo_x86.txt
set print elements 0
set print repeats 0
set pagination off
break gdb_anchor
run
n
n
n
set logging file demo_x86.txt
set logging on
print /d *output@output_size
set logging off
quit
