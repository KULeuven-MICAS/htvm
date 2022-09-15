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

def preprocess_cycle_counts(names, cycle_counts):
    keys = names
    values = cycle_counts
    total_dory = 0
    total_tvm = 0
    def is_dory_function(name):
        return "soma_dory_main" in name
    for name, cycles in zip(keys,values):
        if is_dory_function(name):
            total_dory += cycles
        else:
            total_tvm += cycles
    return total_dory, total_tvm, total_dory+total_tvm

def bar_plot_total(names, cycle_counts, relative=True):
    dory_list = []
    tvm_list = []
    opt_level = ["O0","O1","O2","O3"]
    for cycles in cycle_counts:
        dory, tvm,  tot = preprocess_cycle_counts(names, cycles)
        if relative:
            dory_list.append(dory/tot)
            tvm_list.append(tvm/tot)
        else:
            dory_list.append(dory)
            tvm_list.append(tvm)
    if relative:
        text_template = "%{y}"
    else:
        text_template = None
    data = [
            go.Bar(
                   x=opt_level,
                   y=dory_list,
                   name="Dory",
                   text=dory_list,
                   texttemplate=text_template
                  ),
             go.Bar(
                    x=opt_level,
                    y=tvm_list,
                    name="TVM",
                    text=tvm_list,
                    texttemplate=text_template
                    )
             ]
    fig = go.Figure(data=data)
    y_title = "Cycles"
    if relative:
        y_title = "Cycles (%)"
        fig.update_yaxes(tickformat=".2%", range=[0,1])
    fig.update_layout(title="Relay Resnet20 Backend Comparison",
                      barmode="relative", font_family="Droid Sans Mono",
                      xaxis_title="GCC Optimization Level", yaxis_title=y_title,
                      legend_title="Backend")
    fig.write_html("bar_plot_total.html", auto_open=True)


def bar_plot_individual(names, cycle_counts, log=True):
    def get_opacity(name):
        if "soma_dory_main" in name:
            return 1
        else:
            return 0.5

    # ! bar chart is horizontal
    data = []
    for i, opt_level in enumerate(["O0","O1","O2","O3"]):
    # Use customdata to access full names on hover
        data.append(go.Bar(
                           x=cycle_counts[i],
                           y=make_names_unique(names),
                           customdata=names,
                           name=opt_level,
                           orientation="h",
                           marker={'opacity': [get_opacity(name) for name in names]}
                           ))
    # Specify autorange reversed to put first executed kernel at top
    # (Default is to put first executed kernel at bottom)
    fig = go.Figure(data=data,
                    layout=go.Layout(
                              yaxis=dict(
                                  autorange='reversed'
                                  )
                              ),
                    )
    x_title = "Cycles"
    if log:
        fig.update_xaxes(type="log")
        x_title = "Cycles (Log)"
    fig.update_layout(title="Relay Resnet20 Cycle Rundown",
                      barmode="group", font_family="Droid Sans Mono",
                      xaxis_title=x_title, yaxis_title="Kernel",
                      legend_title="GCC Optimization Level")
    # Add full name on hover
    fig.update_traces(hovertemplate = "<b>full name:</b> %{customdata}<br><b>cycles:</b>     %{x}")
    # Shorten display names
    fig.update_layout(yaxis={
                        'tickmode': 'array',
                        'tickvals': list(range(len(names))),
                        'ticktext': preprocess_names(names),
                        },
                     )
    fig.write_html("bar_plot.html", auto_open=True)


def make_names_unique(names):
    # This is often necessary to make sure plotly doesn't
    # Stack the wrong data on top of each other
    names = [f"{i}_{name}" for i, name in enumerate(names)]


if __name__ == "__main__":
    names, cycle_counts_O0 = get_values("results/relay_resnet20_dory_fused_O0_individual.csv")
    names, cycle_counts_O1 = get_values("results/relay_resnet20_dory_fused_O1_individual.csv")
    names, cycle_counts_O2 = get_values("results/relay_resnet20_dory_fused_O2_individual.csv")
    names, cycle_counts_O3 = get_values("results/relay_resnet20_dory_fused_O3_individual.csv")
    cycle_counts = [cycle_counts_O0, cycle_counts_O1, cycle_counts_O2, cycle_counts_O3]
    bar_plot_total(names, cycle_counts, relative=False)
    bar_plot_individual(names, cycle_counts, log=True)
    
