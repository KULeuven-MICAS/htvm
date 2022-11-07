import re
import glob
import csv


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
    re_kernel = r"(if \(tvmgen_default_(\w*(_\d*)?)\(.*\) return -1;)"
    kernels = []
    def add_perf_counter(matchobj):
        """
        This function scans for tvm kernels, adds in performance counters,
        and adds the name of the kernel to a list
        """
        nonlocal perf_counter_no
        setup = "init_global_perf_counter();\n"
        before = "start_perf_counter();\n"
        after = f"perf_cyc_tvm_{perf_counter_no} = stop_perf_counter();\n"
        # Use perf_counter_no from outside to see if this
        # is the first replacement
        nonlocal first_replacement
        kernels.append(f"tvmgen_default_{matchobj[2]}")
        if first_replacement:
            first_replacement = False
            perf_counter_no += 1
            return setup + "  " + before + "  " + matchobj[0] + "\n  " + after
        else:
            perf_counter_no += 1
            return before + "  " + matchobj[0] + "\n  "+ after
    result = re.sub(re_kernel, add_perf_counter,
                    code_string, count=0, flags=re.MULTILINE)
    return result, kernels


def generate_gdb_script(kernel_counters, logging_file="profile.txt",
                        measurement="individual"):
    """
    This code will generate gdb script which will print every performance
    counter by stepping after tvmgen_default_run and then printing out
    the global variables.
    """
    preamble = \
        f"set logging file {logging_file}\n" + \
        "set logging on\n"
    body = ""
    if measurement == "individual":
        for kernel_counter in kernel_counters:
            body += f"print {kernel_counter}\n"
    else:
    # global measurement
        body += "print perf_cyc\n"
    closing = "set logging off\n" + \
        "quit\n"
    return preamble + body + closing


def parse_gdb_log(file_name="profile.txt"):
    with open(file_name, "r") as log_file:
        log = log_file.read()
        gdb_regex = r"\$\d* = (\d*)"
        entries = re.finditer(gdb_regex, log, flags=re.MULTILINE)
        # Return only the cycles, not the entire regex match
        return [int(i[1]) for i in entries]

def add_headers(code_string, kernel_counters):
    """
    This function alters default_lib1.c, it:
    1) includes pulp.h "#include pulp.h"
    2) includes pulp.h "#include pulp.h"
    3) globally declares tvm_kernel_counters

    The function returns the altered code string
    """
    # declare counter stores for all kernels
    global_counter_decl = ""
    for kernel_counter in kernel_counters:
        global_counter_decl += f"volatile int {kernel_counter};\n"
    """
    This regex matches include statements in the beginning of a C file e.g.

    #include "tvm/runtime/c_runtime_api.h"
    #include "tvm/runtime/c_backend_api.h"
    #include <math.h>

    Only replace one time, hence count = 1
    """
    re_header = r"(#include (?:\"|<).*(?:\"|>))"
    sub_header = "\\1\n#include \"pulp.h\"\n" + \
                 "#include \"pulp_rt_profiler_wrapper.h\"\n" + \
                 global_counter_decl
    replaced_code_string = re.sub(re_header, sub_header, code_string, count=1,
                                  flags=re.MULTILINE)
    return replaced_code_string


class DianaResult():
    # kernel names are supposed to be stored in calling order.
    # If layer is called multiple times, it has to be duplicated.
    kernel_names = []
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
        results = iter(self.results_string)
        for name in self.kernel_names:
            if self.is_dory_kernel(name):
                cycles = next(results)
                dory_total += cycles
                total += cycles
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

    @staticmethod
    def is_dory_kernel(kernel_name):
        regex_function = r"tvmgen_default_soma_dory_main_(\d*)"
        match_object = re.search(regex_function, kernel_name,
                                 flags=re.MULTILINE)
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
                cycles = next(results)
                writer.writerow([name, cycles])


def adapt_lib1(file_name):
    """
    Adds performance counters in default_lib1.c to measure
    individual kernel performance

    file_name: (full) path to default_lib1.c
    tvm_kernel_counters: list of kernel counters to be inserted

    Note: Dory kernel counters are replaced in place.
    """
    re_main = r"TVM_DLL int32_t tvmgen_default___tvm_main__\(.*\) {((.|\n)*)}"
    with open(file_name, "r+") as lib1:
        data = lib1.read()
        failsafe_check(data)
        # Update all TVM generated code in default_lib1.c
        main_function_string = re.search(re_main, data, flags=re.MULTILINE)[0]
        replaced_code, kernels = add_tvm_test_code_in_main(main_function_string)
        # Replace the main function call with added perf counters
        replaced_script = re.sub(re_main, replaced_code, data, count=0,
                                 flags=re.MULTILINE)
        # Generate kernel counter strings for header and gdb_script
        counters = [f"perf_cyc_tvm_{i}" for i,kernel in enumerate(kernels)]
        # Change system header, add kernel counters here as globals
        replaced_script = add_headers(replaced_script, counters)
        # Seek and truncate are necessary for overwriting the read file
        lib1.seek(0)
        lib1.write(replaced_script)
        lib1.truncate()
        print(f"Updated main file @ {file_name}")
        return counters, kernels


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
                          '#include "pulp_rt_profiler_wrapper.h"\n',
                          data, count=0, flags=re.MULTILINE)
        decl = "volatile int perf_cyc;\n"
        setup = "init_global_perf_counter();\n"
        before = "start_perf_counter();\n"
        after = "perf_cyc = stop_perf_counter();\n"
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


def insert_profiler(codegen_dir="./build/codegen/host/src/",
                    gdb_script_name="./gdb_demo.sh",
                    gdb_log_name="./profile.txt",
                    csv_file="profile.csv",
                    interactive=False,
                    measurement="individual"):
    # Skip early in this case
    if measurement is None or measurement == "power":
        return
    lib1_file_name = codegen_dir + "default_lib1.c"
    lib0_file_name = codegen_dir + "default_lib0.c"
    # Providing empty kernel_counters in case default_lib1.c is not updated
    kernel_counters = None
    # Update default_lib1.c
    if measurement == "individual":
        kernel_counters, kernels = adapt_lib1(lib1_file_name)
    if measurement == "global":
        adapt_lib0(lib0_file_name)
    # Write test script which goes with this file
    print(f"Appending perf counters to GDB script ({gdb_script_name})")
    # Append to the existing gdb script
    with open(gdb_script_name, "a") as gdb_script:
        gdb_script.write(generate_gdb_script(kernel_counters, gdb_log_name,
                                             measurement=measurement))
    if interactive and measurement is not None:
        print("Ready for parsing GDB output")
        print("Please run to start profiling on Diana")
        input("Press enter after profiling run...")
        log_results = parse_gdb_log()
        if measurement == "individual":
            result = DianaResult(kernels, log_results)
            print("\n-----  RESULTS ------")
            result.pretty_print()
            print("\n")
            result.print_total_cycles()
            print(f"\nExporting CSV results to \"{csv_file}\", exiting")
            result.write_csv(csv_file)
        else:
            # global measurement
            global_cycles = log_results[0]
            print("\n-----  GLOBAL RESULT ------")
            print(f"Total cycles  {global_cycles:12,} (100%)\n")
            print(f"Exporting CSV results to \"{csv_file}\", exiting")
            with open(csv_file, "w") as csv:
                csv.write(f"{csv_file}_total,{global_cycles}\n")
    else:
        # non interactive mode
        print("Files are updated, gdb script is generated, exiting")


if __name__ == "__main__":
    insert_profiler()
