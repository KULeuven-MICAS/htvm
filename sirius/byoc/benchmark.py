import re
import pathlib
import datetime
import glob

"""
Regex explanation:

RE_TVM regex matches C code lines like:

if (tvmgen_default_fused_add_3(sid_2, sid_3, sid_1) != 0 ) return -1;

but not dory-generated like:

if (tvmgen_default_soma_dory_main_3(sid_1, sid_2) != 0 ) return -1;

The latter are detected by RE_DORY regex.
RE_BOTH matches both.

It can be used to add performance counter code to TVM generated C kernels
Since the dory generated C kernels already have more finegrained performance
counter implementations.
"""

RE_TVM = r"(if \(tvmgen_default_(?!soma_dory_main_\d*)(\w*(_\d*)?)\(.*\)" + \
        " return -1;)"
RE_DORY = r"(if \((tvmgen_default_soma_dory_main_(\d*)?)?\(.*\) return -1;)"
RE_BOTH = r"(if \(tvmgen_default_(\w*(_\d*)?)\(.*\) return -1;)"

"""
This regex matches the tvmgen_default___tvm_main(*){} function
NOTE: This regex might not work properly if the function is not the last
one in the file!
"""

RE_MAIN = r"TVM_DLL int32_t tvmgen_default___tvm_main__\(.*\) {((.|\n)*)}"

"""
This regex matches include statements in the beginning of a C file e.g.

#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
"""

RE_INC = r"(#include (?:\"|<).*(?:\"|>))"


def add_tvm_test_code_in_main(code_string: str):
    """
    This function searches with regex for lines like
    returns: list of all functions detected in main
    """
    # With dory functions
    # Without dory functions

    # Use this to only insert setup logic in first run of this algo
    first_replacement = True
    perf_counter_no = 0

    def add_perf_counter(matchobj):
        nonlocal perf_counter_no
        setup = \
                        "  perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));\n" + \
            "  rt_perf_init(perf);\n" + \
            "  rt_perf_conf(perf, (1<<RT_PERF_CYCLES));\n  "
        before = \
            "rt_perf_stop(perf);\n" + \
            "  rt_perf_start(perf);\n"
        after = \
            "  rt_perf_stop(perf);\n" + \
            "  rt_perf_save(perf);\n" + \
            f"  perf_cyc_tvm_{perf_counter_no} = rt_perf_get(perf, RT_PERF_CYCLES);\n" + \
            "  rt_perf_reset(perf);"
        # Use perf_counter_no from outside to see if this is the first replacement
        nonlocal first_replacement
        if first_replacement:
            first_replacement = False
            perf_counter_no += 1
            return setup + before + "  " + matchobj[0] + "\n" + after
        else:
            perf_counter_no += 1
            return before + "  " + matchobj[0] + "  \n" + after
    result = re.sub(RE_TVM, add_perf_counter,
                    code_string, count=0, flags=re.MULTILINE)
    return result


def generate_gdb_script(tvm_kernels, logging_file="benchmark.txt"):
    """
    This code will generate gdb script which will print every performance
    counter by stepping after tvmgen_default_run and then printing out
    the global variables.
    """
    preamble = \
        "!rm benchmark.txt\n" + \
        "set print elements 0\n" + \
        "set print repeats 0\n" + \
        "set pagination off\n" + \
        "file build/pulpissimo/demo/demo\n" + \
        "target remote localhost:3333\n" + \
        "load\n" + \
        "break tvmgen_default_run\n" + \
        f"set logging file {logging_file}\n" + \
        "set logging on\n" + \
        "\n" + \
        "c\n" + \
        "n\n"
    body = ""
    closing = \
        "set logging off\n" + \
        "quit\n"
    for kernel_no in range(len(tvm_kernels)):
        body += f"print perf_cyc_tvm_{kernel_no}\n"
    return preamble + body + closing


def parse_gdb_log(tvm_kernels, file_name = "benchmark.txt"):
    with open(file_name, "r") as log_file:
        log = log_file.read()
        gdb_regex = r"\$\d* = (\d*)"
        entries = re.finditer(gdb_regex, log, flags=re.MULTILINE) 
        # Return dictionary which contains names and cycle counts
        return {tvm_kernels[i]:e[1] for i,e in enumerate(entries)}



def get_kernels(main_function):
    tvm_kernels = ["tvmgen_default_" + i[2] for i in
                   re.finditer(RE_TVM, main_function, flags=re.MULTILINE)]
    dory_kernels = [i[2] for i in
                    re.finditer(RE_DORY, main_function, flags=re.MULTILINE)]
    all_kernels = ["tvmgen_default_" + i[2] for i in
                   re.finditer(RE_BOTH, main_function, flags=re.MULTILINE)]
    return tvm_kernels, dory_kernels, all_kernels


def add_headers(code_string, tvm_kernels):
    no_of_counters = len(tvm_kernels)
    # Add global declaration of perf structure 
    global_counter_decl = "volatile rt_perf_t *perf;\n"
    # Also declare counter stores for all kernels
    for i in range(no_of_counters):
        global_counter_decl += f"int perf_cyc_tvm_{i};\n"
    # Add <<#include "pulp.h">> (only once, hence count=1)
    replaced_code_string  = re.sub(RE_INC, "\\1\n#include \"pulp.h\"\n" + \
                                   global_counter_decl,
                                   code_string, count=1, flags=re.MULTILINE)
    return replaced_code_string

def replace_dory_declarations(code_string):
    # Remove declarations here, since they were moved to default_lib1.c
    regex_decl = r"\/\/ perf measurement begin\n" + \
                 r"\s*volatile rt_perf_t \*perf;\n" + \
                 r"\s*perf = rt_alloc\(RT_ALLOC_L2_CL_DATA, sizeof\(rt_perf_t\)\);"
    regex_init = r"int perf_cyc, perf_cyc1, perf_cyc2;\n" + \
                 r"\s*rt_perf_init\(perf\);\n" + \
                 r"\s*rt_perf_conf\(perf, \(1<<RT_PERF_CYCLES\)\);"
    replaced = re.sub(regex_decl, "", code_string, count=1, flags=re.MULTILINE)
    replaced = re.sub(regex_init, "", replaced, count=1, flags=re.MULTILINE)
    # Change the names of perf_cyc etc to match their function name
    # First extract the function name
    regex_function = r"int32_t (tvmgen_default_soma_dory_main_(\d*))(.*)"
    function_search  = re.search(regex_function, replaced, flags=re.MULTILINE)
    # E.g. for "tvmgen_default_soma_dory_main_57", extract "57"
    function_number = function_search[2]
    # Matches "perf_cyc", but not perf_cyc2, perf_cyc1
    regex_setup = r"perf_cyc(?!1|2)"
    subst_setup = f"perf_setup_{function_number}"
    replaced = re.sub(regex_setup, subst_setup, replaced, 
                      count=0, flags=re.MULTILINE)
    regex_calc = r"perf_cyc1"
    subst_calc = f"perf_calc_{function_number}"
    replaced = re.sub(regex_calc, subst_calc, replaced, 
                      count=0, flags=re.MULTILINE)
    regex_retr = r"perf_cyc2"
    subst_retr = f"perf_retr_{function_number}"
    replaced = re.sub(regex_retr, subst_retr, replaced, 
                      count=0, flags=re.MULTILINE)

    return replaced

def update_dory_default_libs(codegen_dir):
    # Find all default_libx.c files
    glob_pattern = codegen_dir + "default_lib*.c"
    default_libs = glob.glob(glob_pattern)
    # remove default_lib0.c and default_lib1.c
    default_libs.remove(codegen_dir+"default_lib0.c")
    default_libs.remove(codegen_dir+"default_lib1.c")
    for default_lib in default_libs:
        with open(default_lib, "r+") as dory_lib:
            data = dory_lib.read()
            replaced = replace_dory_declarations(data)
            dory_lib.write(replaced)
            dory_lib.seek(0)
            dory_lib.write(replaced)
            dory_lib.truncate()
            print(f"Updated {default_lib}")


if __name__ == "__main__":
    verbose = False
    file_name = "./build/codegen/host/src/default_lib1.c"
    # Test code for dory adaptions
    codegen_dir = "./build/codegen/host/src/"
    # Update test code for dory files
    update_dory_default_libs(codegen_dir)
    gdb_script_name = "./gdb_benchmark.sh"
    with open(file_name, "r+") as lib1:
        data = lib1.read()
        check = re.search("#include \"pulp.h\"", data, flags=re.MULTILINE)
        if check != None:
            raise RuntimeError("You've already run the script, not updating")
        main_function_string = re.search(RE_MAIN, data, flags=re.MULTILINE)[0]
        tvm_ks, dory_ks, all_ks = get_kernels(main_function_string)
        replaced_code = add_tvm_test_code_in_main(main_function_string)
        # Write test script which goes with this file
        with open(gdb_script_name, "w") as gdb_script:
            gdb_script.write(generate_gdb_script(tvm_ks,"benchmark.txt"))


        # Replace the main function call with added perf counters
        replaced_script = re.sub(RE_MAIN, replaced_code, data, count=0,
                                 flags=re.MULTILINE)
        replaced_script = add_headers(replaced_script, tvm_ks)
        # Seek and truncate are necessary for overwriting the read file
        lib1.seek(0)
        lib1.write(replaced_script)
        lib1.truncate()
        print(f"Updated main file @ {file_name}")
    input("Ready for parsing GDB output, please start the benchmark on diana")
    results = parse_gdb_log(tvm_ks)
    print(f"Results")
    print(f"=======")
    for i, (name, cycles) in enumerate(results.items()):
        print(f"{i}) {name: <50} : {int(cycles):,}")
