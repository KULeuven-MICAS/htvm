void __attribute__((noinline, optimize("O0"))) gdb_anchor(){
    // This function doesn't do anything.
    // But it can be used to insert breakpoints in optimized code.
    return;
}
