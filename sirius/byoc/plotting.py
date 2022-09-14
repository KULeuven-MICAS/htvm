import matplotlib.pyplot as plt
import numpy as np
import csv


def get_values(path="results/relay_resnet20_dory_fused_O3_individual.csv"):
    with open(path, "r") as results:
        names = []
        cycle_counts = []
        results_reader = csv.reader(results)
        for i in list(results_reader):
            names.append(i[0])
            cycle_counts.append(int(i[1]))
        return names, cycle_counts

def preprocess_names(names):
    def shorten_long_name(long_name):
        # Set limit of characters
        limit = 23
        if len(long_name) > limit:
            beginning = long_name[:10]
            ending = long_name[-10:]
            return beginning + "..." + ending
        else:
            return long_name

    names = [name.replace("tvmgen_default_","") for name in names]
    names = [shorten_long_name(name) for name in names]
    return names

def bar_plot(names, cycle_counts):
    names = preprocess_names(names)
    data = ((3, 10, 3), (10, 3, 2), (10, 3, 3), (5, 8, 5), (5, 1, 6))
    #data = ((3, 1000), (10, 3), (100, 30), (500, 800), (50, 1))
    data = cycle_counts
    dim = len(data[0])
    width = 0.20

    fig, ax = plt.subplots()
    x = np.arange(len(data))
    for i in range(len(data[0])):
        y = [d[i] for d in data]
        print(y)
        b = ax.bar(x + width * i, height=y, width=width, bottom=0.001)
        
    ax.set_xticks(x + width, labels=names, rotation=90)
    ax.set_ylim(10, 10**7)
    ax.set_yscale('log')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(["O0","O1","O2","O3"])
    ax.grid(axis = 'y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    names, cycle_counts_O0 = get_values("results/relay_resnet20_dory_fused_O0_individual.csv")
    names, cycle_counts_O1 = get_values("results/relay_resnet20_dory_fused_O1_individual.csv")
    names, cycle_counts_O2 = get_values("results/relay_resnet20_dory_fused_O2_individual.csv")
    names, cycle_counts_O3 = get_values("results/relay_resnet20_dory_fused_O3_individual.csv")
    cycle_counts = list(zip(cycle_counts_O0, cycle_counts_O1, cycle_counts_O2, cycle_counts_O3))
    bar_plot(names, cycle_counts)
    





