file my_build/helloworld_gdb
break gdb_anchor
run
# At this point you will hit the gdb_anchor breakpoint
# Now you can print the string
printf "%s", (char *)global_string
continue
quit