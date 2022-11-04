import utils
import numpy as np

if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    demo_result = utils.get_gdb_output("demo.txt")
    demo_x86_result = utils.get_gdb_output("demo_x86.txt")
    if(np.ma.allequal(demo_result,demo_x86_result)):
        print("SUCCESS: x86 and Diana values are the same")
    else:
        print("FAIL: x86 and Diana values are NOT the same")
        print(demo_result - demo_x86_result)
