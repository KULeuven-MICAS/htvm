!rm gdb.txt
file build/pulpissimo/demo/demo
target remote localhost:3333
load
break main
c
