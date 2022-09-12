import re
import glob
import csv

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
        setup = "init_global_perf_counter();\n"
        before = "start_benchmark();\n"
        after = f"perf_cyc_tvm_{perf_counter_no} = stop_benchmark();\n"
        # Use perf_counter_no from outside to see if this
        # is the first replacement
        nonlocal first_replacement
        if first_replacement:
            first_replacement = False
            perf_counter_no += 1
            return setup + "  " + before + "  " + matchobj[0] + "\n  " + after
        else:
            perf_counter_no += 1
            return before + "  " + matchobj[0] + "\n  "+ after
    result = re.sub(RE_BOTH, add_perf_counter,
                    code_string, count=0, flags=re.MULTILINE)
    return result


def generate_gdb_script(kernel_counters, logging_file="benchmark.txt",
                        measurement="individual"):
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
        "break gdb_anchor\n" + \
        f"set logging file {logging_file}\n" + \
        "set logging on\n"
    body = "c\n" + "n\n"
    if measurement == "individual":
        for kernel_counter in kernel_counters:
            body += f"print {kernel_counter}\n"
    else:
    # global measurement
            "print perf_cyc\n"
    closing = "set logging off\n" + \
        "quit\n"
    return preamble + body + closing


def generate_kernel_counters(all_kernels, only_tvm=False):
    kernel_counters = []
    tvm_counter = 0
    for kernel in all_kernels:
        # check if the kernel is TVM- or DORY-generated
        regex_function = r"tvmgen_default_soma_dory_main_(\d*)"
        match_object = re.search(regex_function, kernel, flags=re.MULTILINE)
        if not only_tvm and (match_object is not None):
            function_number = match_object[1]
            kernel_counters.append(f"perf_setup_{function_number}")
            kernel_counters.append(f"perf_calc_{function_number}")
            kernel_counters.append(f"perf_retr_{function_number}")
        else:
            kernel_counters.append(f"perf_cyc_tvm_{tvm_counter}")
            tvm_counter += 1
    return kernel_counters


def parse_gdb_log(file_name="benchmark.txt"):
    with open(file_name, "r") as log_file:
        log = log_file.read()
        gdb_regex = r"\$\d* = (\d*)"
        entries = re.finditer(gdb_regex, log, flags=re.MULTILINE)
        # Return only the cycles, not the entire regex match
        return [int(i[1]) for i in entries]


def get_kernels(main_function):
    tvm_kernels = ["tvmgen_default_" + i[2] for i in
                   #re.finditer(RE_TVM, main_function, flags=re.MULTILINE)]
                   re.finditer(RE_BOTH, main_function, flags=re.MULTILINE)]
    dory_kernels = [i[2] for i in
                    re.finditer(RE_DORY, main_function, flags=re.MULTILINE)]
    all_kernels = ["tvmgen_default_" + i[2] for i in
                   re.finditer(RE_BOTH, main_function, flags=re.MULTILINE)]
    return tvm_kernels, dory_kernels, all_kernels


def add_headers(code_string, tvm_kernel_counters):
    """
    This function alters default_lib1.c, it:
    1) includes pulp.h "#include pulp.h"
    2) includes pulp.h "#include pulp.h"
    3) globally declares tvm_kernel_counters

    The function returns the altered code string
    """
    # declare counter stores for all kernels
    global_counter_decl = ""
    for kernel_counter in tvm_kernel_counters:
        global_counter_decl += f"volatile int {kernel_counter};\n"
    # Add <<#include "pulp.h">> (only once, hence count=1)
    replaced_code_string = re.sub(RE_INC, "\\1\n#include \"pulp.h\"\n#include \"pulp_rt_benchmark_wrapper.h\"\n" +
                                  global_counter_decl, code_string,
                                  count=1, flags=re.MULTILINE)
    return replaced_code_string


def remove_dory_declarations(code_string):
    """
    Function that for each dory-generated kernel:

    1) removes perf declarations and performance counters
       (The one in default_lib1.c)

    This function returns the altered string representation of the file
    """
    # Remove declarations here, since they were moved to default_lib1.c
    regex_decl = r"\/\/ perf measurement begin\n" + \
                 r"\s*volatile rt_perf_t \*perf;\n" + \
                 r"\s*perf = rt_alloc\(RT_ALLOC_L2_CL_DATA, " + \
                 r"sizeof\(rt_perf_t\)\);"
    # The perf pointer is declared in default_lib1.c
    # so it has to be declared as extern
    replaced = re.sub(regex_decl, "", code_string, count=1, flags=re.MULTILINE)
    # These declarations are not valid anymore
    regex_init = r"\s*int perf_cyc, perf_cyc1, perf_cyc2;\n"
    replaced = re.sub(regex_init, "", replaced, count=1,
                      flags=re.MULTILINE)
    regex_start = r"\s*rt_perf_stop\(perf\);\n\s*rt_perf_save\(perf\);" + \
        r"\n\s*perf_cyc\d* = rt_perf_get\(perf, RT_PERF_CYCLES\);\n" + \
        r"\s*rt_perf_reset\(perf\);\n"
    replaced = re.sub(regex_start, "", replaced, count=0,
                      flags=re.MULTILINE)
    regex_reset = r"\s*rt_perf_init\(perf\);\n" + \
        r"\s*rt_perf_conf\(perf, \(1<<RT_PERF_CYCLES\)\);\n" + \
        r"\s*rt_perf_stop\(perf\);\n\s*rt_perf_start\(perf\);\n"
    replaced = re.sub(regex_reset, "", replaced, count=0,
                      flags=re.MULTILINE)

    return replaced


def replace_dory_declarations(code_string):
    """
    Function that for each dory-generated kernel:

    1) removes perf declarations and references external one
       (The one in default_lib1.c)
    2) replaces perf_cyc,perf_cyc1 and perf_cyc2 by
       perf_setup_x, perf_calc_x and perf_retr_x where x marks
       the number of the generated dory function:
       E.g. for "tvmgen_default_soma_dory_main_57", x = 57
    3) declares perf_setup_x, perf_calc_x and perf_retr_x globally
       in the given file.

    This function returns the altered string representation of the file
    """
    # Remove declarations here, since they were moved to default_lib1.c
    regex_decl = r"\/\/ perf measurement begin\n" + \
                 r"\s*volatile rt_perf_t \*perf;\n" + \
                 r"\s*perf = rt_alloc\(RT_ALLOC_L2_CL_DATA, " + \
                 r"sizeof\(rt_perf_t\)\);"
    # The perf pointer is declared in default_lib1.c
    # so it has to be declared as extern
    replaced = re.sub(regex_decl, "extern rt_perf_t *perf;",
                      code_string, count=1, flags=re.MULTILINE)
    # These declarations are not valid anymore
    regex_init = r"int perf_cyc, perf_cyc1, perf_cyc2;\n"
    subst_init = "rt_perf_reset(perf);\n"
    replaced = re.sub(regex_init, subst_init, replaced, count=1,
                      flags=re.MULTILINE)
    # Change the names of perf_cyc etc to match their function name
    # First extract the function name
    regex_function = r"int32_t (tvmgen_default_soma_dory_main_(\d*))(.*)"
    function_search = re.search(regex_function, replaced, flags=re.MULTILINE)
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
    # insert global variables here
    global_counter_decl = "int " + subst_setup + ";\n"
    global_counter_decl += "int " + subst_calc + ";\n"
    global_counter_decl += "int " + subst_retr + ";\n"
    replaced = re.sub(r"(include \"dory.h\")",
                      "\\1\n" + global_counter_decl,
                      replaced, count=1, flags=re.MULTILINE)
    return replaced


def update_dory_default_libs(codegen_dir, remove=False):
    """
    Function that goes over all default_libx.c files (except for default_lib0.c
    and default_lib1.c - These are not generated by dory) and calls
    replace_dory_declarations on each file

    codegen_dir is the directory where TVM outputs its codegen

    If the remove flag is used, all the declarations are just removed
    """
    # Find all default_libx.c files
    glob_pattern = codegen_dir + "default_lib*.c"
    default_libs = glob.glob(glob_pattern)
    # remove default_lib0.c and default_lib1.c
    default_libs.remove(codegen_dir+"default_lib0.c")
    default_libs.remove(codegen_dir+"default_lib1.c")
    for default_lib in default_libs:
        with open(default_lib, "r+") as dory_lib:
            data = dory_lib.read()
            if remove:
                replaced = remove_dory_declarations(data)
            else:
                replaced = replace_dory_declarations(data)
            dory_lib.write(replaced)
            dory_lib.seek(0)
            dory_lib.write(replaced)
            dory_lib.truncate()
            if remove:
                print(f"Removed dory decl in {default_lib}")
            else:
                print(f"Updated {default_lib}")


class DianaResult():
    # kernel names are supposed to be stored in calling order.
    # If layer is called multiple times, it has to be duplicated.
    kernel_names = []
    # results strings are expected to be stored in the same order
    # TVM  result --> one cycle count (no setup and retr)
    # DORY result --> three cycle counts (setup, calc, and retr)
    results_string = []

    def __init__(self, kernels, gdb_log):
        self.kernel_names = kernels
        self.results_string = gdb_log

    def __get_longest_name(self):
        """
        Gets the first occurence of the longest name in self.kernel_names
        """
        longest_length = 0
        longest_string = ""
        for name in self.kernel_names:
            if len(name) > longest_length:
                longest_length = len(name)
                longest_string = name
        return longest_string

    def print_total_cycles(self):
        total = 0
        tvm_total = 0
        dory_total = 0
        dory_setup_total = 0
        dory_calc_total = 0
        dory_retr_total = 0
        results = iter(self.results_string)
        for name in self.kernel_names:
            if self.is_dory_kernel(name):
                setup = next(results)
                calc = next(results)
                retr = next(results)
                total += setup + calc + retr
                dory_total += setup + calc + retr
                dory_setup_total += setup
                dory_calc_total += calc
                dory_retr_total += retr
            else:
                cycles = next(results)
                total += cycles
                tvm_total += cycles
        print("CYCLE RUNDOWN")
        print(f"Total cycles  {total:12,} (100%)")
        print(f"- TVM cycles  {tvm_total:12,} (" +
              f"{100 * tvm_total/total:3.1f}%)")
        print(f"- DORY cycles {dory_total:12,} (" +
              f"{100 * dory_total/total:3.1f}%)")
        print(f"--- setup     {dory_setup_total:12,} (" +
              f"{100 * dory_setup_total/total:3.1f}%)")
        print(f"--- calculate {dory_calc_total:12,} (" +
              f"{(100 * dory_calc_total/total):3.1f}%)")
        print(f"--- retrieve  {dory_retr_total:12,} (" +
              f"{(100 * dory_retr_total/total):3.1f}%)")

    @staticmethod
    def is_dory_kernel(kernel_name):
        regex_function = r"tvmgen_default_soma_dory_main_(\d*)"
        match_object = re.search(regex_function, kernel_name,
                                 flags=re.MULTILINE)
        return False
        if match_object is not None:
            return True
        else:
            return False

    def pretty_print(self):
        """
        Pretty print results
        """
        results = iter(self.results_string)
        offset = len(self.__get_longest_name())
        separator = (offset + 30) * "-"
        for i, name in enumerate(self.kernel_names):
            if self.is_dory_kernel(name):
                setup = next(results)
                calc = next(results)
                retr = next(results)
                print(separator)
                print(f"{i:<3}) {name: <{offset}} : setup : {int(setup):,}")
                empty = ""
                print(f"     {empty:<{offset}} : calc  : {int(calc):,}")
                print(f"     {empty:<{offset}} : retr  : {int(retr):,}")
            else:
                cycles = next(results)
                print(separator)
                print(f"{i:<3}) {name: <{offset}}         : {int(cycles):,}")

    def write_csv(self, file_name="results.csv"):
        """
        Write results to CSV file
        """
        results = iter(self.results_string)
        with open(file_name, "w") as csv_file:
            writer = csv.writer(csv_file)
            for name in self.kernel_names:
                if self.is_dory_kernel(name):
                    setup = next(results)
                    calc = next(results)
                    retr = next(results)
                    writer.writerow([name + "_setup", setup])
                    writer.writerow([name + "_calc", calc])
                    writer.writerow([name + "_retr", retr])
                else:
                    cycles = next(results)
                    writer.writerow([name, cycles])


def adapt_lib1(file_name, tvm_kernel_counters):
    """
    Adds performance counters in default_lib1.c to measure
    individual kernel performance

    file_name: (full) path to default_lib1.c
    tvm_kernel_counters: list of kernel counters to be inserted

    Note: Dory kernel counters are replaced in place.
    """
    with open(file_name, "r+") as lib1:
        data = lib1.read()
        # Update all TVM generated code in default_lib1.c
        main_function_string = re.search(RE_MAIN, data, flags=re.MULTILINE)[0]
        replaced_code = add_tvm_test_code_in_main(main_function_string)
        # Replace the main function call with added perf counters
        replaced_script = re.sub(RE_MAIN, replaced_code, data, count=0,
                                 flags=re.MULTILINE)
        # Change system header, only add tvm_kernel counters here as globals
        replaced_script = add_headers(replaced_script, tvm_kernel_counters)
        # Seek and truncate are necessary for overwriting the read file
        lib1.seek(0)
        lib1.write(replaced_script)
        lib1.truncate()
        print(f"Updated main file @ {file_name}")


def scan_lib1(file_name):
    with open(file_name, "r") as lib1:
        data = lib1.read()
        failsafe_check(data)
        main_function_string = re.search(RE_MAIN, data, flags=re.MULTILINE)[0]
        tvm_ks, dory_ks, all_ks = get_kernels(main_function_string)
        kernel_counters = generate_kernel_counters(all_ks)
        # Generate kernel counter strings for header and gdb_script
        tvm_kernel_counters = generate_kernel_counters(all_ks,
                                                       only_tvm=True)
        return kernel_counters, tvm_kernel_counters, all_ks


def failsafe_check(data):
    """
    Failsafe, check whether default_lib0.c or default_lib1.c
    was already altered

    data : string of input file (default_lib0.c or default_lib1.c)
    """
    check = re.search("#include \"pulp.h\"", data, flags=re.MULTILINE)
    if check is not None:
        raise RuntimeError("You've already run the script, not updating")


def adapt_lib0(file_name):
    """
    Adds performance counter in default_lib0.c to measure overall performance
    as opposed to adding performance counter per invoked kernel in
    default_lib1.c

    file_name: (full) path to default_lib0.c
    """
    with open(file_name, "r+") as lib0:
        data = lib0.read()
        failsafe_check(data)
        # Add "#include "pulp.h""
        replaced = re.sub(r"(#include \<tvmgen_default\.h\>)",
                          r"\1\n" + '#include "pulp.h"\n' + \
                          '#include "pulp_rt_benchmark_wrapper.h"\n',
                          data, count=0, flags=re.MULTILINE)
        decl = "volatile int perf_cyc;\n"
        setup = "init_global_perf_counter();\n"
        before = "start_benchmark();\n"
        after = "perf_cyc = stop_benchmark();\n"
        regex = r"(int32_t tvmgen_default_run\(struct " + \
                r"tvmgen_default_inputs\* inputs,struct " + \
                r"tvmgen_default_outputs\* " + \
                r"outputs\) \{)(.*;\n)(})"
        function = r"int status = tvmgen_default___tvm_main__(" + \
                   "inputs->input,outputs->output);\n"
        status = "return status;\n"
        # whitespace
        ws= "    "
        subst = decl + "\n" + r"\1\n" + ws + setup + ws + before + ws + \
                function + ws + after + ws + status + r"\3" 
        replaced = re.sub(regex, subst, replaced, count=0, flags=re.MULTILINE)
        lib0.seek(0)
        lib0.write(replaced)
        lib0.truncate()
        print(f"Updated lib0 file @ {file_name}")


def create_benchmark(codegen_dir="./build/codegen/host/src/",
                     gdb_script_name="./gdb_benchmark.sh",
                     gdb_log_name="./benchmark.txt",
                     csv_file="benchmark.csv",
                     interactive=False,
                     measurement="individual"):
    # Skip early in this case
    if measurement == "no_benchmark":
        return
    lib1_file_name = codegen_dir + "default_lib1.c"
    lib0_file_name = codegen_dir + "default_lib0.c"
    # Measurement can be individual or global
    # measurement = "global"
    # Providing empty kernel_counters in case default_lib1.c is not updated
    k_counters = None
    # Update default_lib1.c
    if measurement == "individual":
        k_counters, tvm_k_counters, all_kernels = scan_lib1(lib1_file_name)
        adapt_lib1(lib1_file_name, tvm_k_counters)
        # Update all Dory generated cycle counters in default_lib*.c
        #update_dory_default_libs(codegen_dir, remove=False)

        # Overriding TVM dory behaviour
        k_counters = tvm_k_counters
        update_dory_default_libs(codegen_dir, remove=True)
    if measurement == "global":
        adapt_lib0(lib0_file_name)
        # Remove all Dory generated cycle counters in default_lib*.c
        update_dory_default_libs(codegen_dir, remove=True)
    # Write test script which goes with this file
    print(f"Generating GDB script ({gdb_script_name})")
    with open(gdb_script_name, "w") as gdb_script:
        gdb_script.write(generate_gdb_script(k_counters, gdb_log_name,
                                             measurement=measurement))
    if interactive and measurement != "no_benchmark":
        print("Ready for parsing GDB output")
        print("Please run the benchmark on Diana")
        input("Press enter after benchmark...")
        log_results = parse_gdb_log()
        if measurement == "individual":
            result = DianaResult(all_kernels, log_results)
            print("\n")
            print("-----  RESULTS ------")
            result.pretty_print()
            print("\n")
            result.print_total_cycles()
            print("\n")
            print(f"Exporting CSV results to \"{csv_file}\", exiting")
            result.write_csv(csv_file)
        else:
            # global measurement
            global_cycles = log_results[0]
            print("\n")
            print("-----  GLOBAL RESULT ------")
            print(f"Total cycles  {global_cycles:12,} (100%)")
            print("\n")
            print(f"Exporting CSV results to \"{csv_file}\", exiting")
            with open(csv_file, "w") as csv:
                csv.write(f"{csv_file}_total, {global_cycles}\n")
    else:
        # non interactive mode
        print("Files are updated, gdb script is generated, exiting")


if __name__ == "__main__":
    create_benchmark()
