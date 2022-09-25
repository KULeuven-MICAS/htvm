import re
import numpy as np

def get_gdb_output(gdb_log_path="debug/gdb.txt"):
    """
    Following lines use the logging output of gdb to match test results with model results

    logging is set by:
        (gdb) set logging on       --> log to gdb.txt
        (gdb) print some_variable
        $1 = \032
        (gdb) set logging off

    In the code below we use regex to match an output created by gdb for an array:
        (gdb) print *my_array@array_size
        $2 = { 69, 420, ... , 42}  --> will be logged to gdb.txt

    After some string manipulation this array is converted to a numpy array.
    This array is checked for a complete match with np.ma.allequal()

    """
    try:
        with open(gdb_log_path) as log:
            data = ""
            for line in log.readlines():
                data += line.strip()
            # Find the right string in gdb log
            matcher = re.compile(r"{.*}",flags=re.DOTALL)
            result = matcher.search(data)
            string = result.group(0)
            # "{ ... , ... }" --> "... , ..."
            string = string.replace("{","")
            string = string.replace("}","")
            # makes a list of numbers in string format
            list_numbers = string.split(",")
            # convert strings to integers
            values = [int(number) for number in list_numbers]
        values_from_test = np.array(values, dtype="int8")
        # Values are returned from GDB in one big one-dimensional tensor
        # Reshaping here such that it matches the output
        return values_from_test
    except FileNotFoundError:
        print("Failed to read out gdb output")
        print(f"Could not find file: {gdb_log_path}")
        print("Did you run the code on gdb?")


if __name__ == "__main__":
    demo_result = get_gdb_output("demo.txt")
    demo_x86_result = get_gdb_output("demo_x86.txt")
    if(np.ma.allequal(demo_result,demo_x86_result)):
        print("SUCCESS: x86 and Diana values are the same")
    else:
        print("FAIL: x86 and Diana values are NOT the same")
