import numpy as np
import csv

import plotly.graph_objects as go

import pandas as pd

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

def bar_plot_plotly(names, cycle_counts):
    names = preprocess_names(names)
    # ! bar chart is horizontal
    data = []
    for i, opt_level in enumerate(["O0","O1","O2","O3"]):
        data.append(go.Bar(
                           x=cycle_counts[i],
                           y=names,
                           name=opt_level,
                           orientation="h",
                           ))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                              yaxis=dict(
                                  autorange='reversed'
                                  )
                              ),
                    )
    fig.update_xaxes(type="log")
    fig.update_layout(barmode="group", font_family="Droid Sans")
    fig.write_html("bar_plot.html", auto_open=True)

if __name__ == "__main__":
    names, cycle_counts_O0 = get_values("results/relay_resnet20_dory_fused_O0_individual.csv")
    names, cycle_counts_O1 = get_values("results/relay_resnet20_dory_fused_O1_individual.csv")
    names, cycle_counts_O2 = get_values("results/relay_resnet20_dory_fused_O2_individual.csv")
    names, cycle_counts_O3 = get_values("results/relay_resnet20_dory_fused_O3_individual.csv")
    cycle_counts = [cycle_counts_O0, cycle_counts_O1, cycle_counts_O2, cycle_counts_O3]
    bar_plot_plotly(names, cycle_counts)
