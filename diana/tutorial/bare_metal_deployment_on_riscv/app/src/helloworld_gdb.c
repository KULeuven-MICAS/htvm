#include "gdb_anchor.h"

char* global_string = "hello world!\n";

int main(int argc(), char* argv[]){
    // We removed the local string here
    // Placing breakpoints on functions makes automated debugging easier.
    gdb_anchor();
}
