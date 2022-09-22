import numpy as np
import csv

import plotly.graph_objects as go
from plotly.io import to_html

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
    
def bar_plot_total_wrapper( data_a, data_b,
                            name_a, name_b,
                            y_title="Cycles",
                            x_categories=["O0","O1","O2","O3"], x_title="GCC Optimization Level",
                            barmode="relative",
                            fig_title=None, legend=True, legend_title=None,
                            percentage=False):
    text_template = "%{y}" if percentage else None
    bar_a = go.Bar(
               x=x_categories,
               y=data_a,
               name=name_a,
               text=data_a,
               texttemplate=text_template,
              )
    if data_b is not None:
        bar_b = go.Bar(
                   x=x_categories,
                   y=data_b,
                   name=name_b,
                   text=data_b,
                   texttemplate=text_template
                  )
        data = [bar_a, bar_b]
    else:
        data = [bar_a]
    fig = go.Figure(data=data)
    if percentage:
        y_title = "Cycles (%)"
        fig.update_yaxes(tickformat=".2%", range=[0,1])
    fig.update_layout(title=fig_title,
                  barmode=barmode, font_family="Droid Sans Mono",
                  xaxis_title=x_title, yaxis_title=y_title,
                  showlegend=legend,
                  legend_title=legend_title, height=600)
    return fig



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
    fig.update_layout(title="Relay Resnet20 Cycle Rundown - GCC Optimization",
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
    return fig


def make_names_unique(names):
    # This is often necessary to make sure plotly doesn't
    # Stack the wrong data on top of each other
    names = [f"{i}_{name}" for i, name in enumerate(names)]

def merge_figures(figures, path="plot.html"):
    html_string = ""
    # Only include JS library the first time
    include_plotlyjs = True
    for figure in figures:
        html_string += to_html(figure, full_html=False, include_plotlyjs=include_plotlyjs)
        include_plotlyjs = False
    with open(path, "w") as plot:
        plot.write(html_string)

def annotate_speedup(figure, values, labels):
    values = cycle_counts_global
    labels = ["O0","O1","O2","O3"]
    diffs = [max(values) - v for v in values]
    speedup = [max(values)/v for v in values]
    diff_labels = dict(zip(labels, diffs))
    i = 0
    for key, value in diff_labels.items():
        if value != 0:
            figure.add_annotation(x=key, y=max(values)-value, ax=0, yanchor='bottom',
                                ay=max(values), ayref='y', showarrow=True,
                                arrowsize=2, arrowhead=1, text=f"-{value} ({speedup[i]:.1f}x)")
        i += 1
    figure.add_traces(go.Scatter(x=labels, y=[max(values)]*len(labels), mode = 'lines',
                   line=dict(color='black', width=1))) 
    return figure


if __name__ == "__main__":
    names = []
    cycle_counts_individual = []
    cycle_counts_global = []
    dory_list = []
    dory_rel_list = []
    tvm_list = []
    tvm_rel_list = []
    tot_list = []
    template = "results_group_x86/relay_groupconv_dory_fused_{}_{}.csv"
    plot_path = "results_group_x86/plot_x86.html"
    for opt_level in ["O0","O1","O2","O3"]:
        # Get individual counts
        names, cycle_counts = get_values(template.format(opt_level,"individual"))
        names = names
        cycle_counts_individual.append(cycle_counts)
        # Get global counts
        dory, tvm, tot = preprocess_cycle_counts(names, cycle_counts)
        tot_list.append(tot)
        dory_rel_list.append(dory/tot)
        tvm_rel_list.append(tvm/tot)
        dory_list.append(dory)
        tvm_list.append(tvm)
        _names, cycle_counts = get_values(template.format(opt_level,"global"))
        cycle_counts_global.append(cycle_counts[0])
    figures = []
    figures.append(bar_plot_individual(names, cycle_counts_individual, log=True))
    figures.append(bar_plot_total_wrapper(data_a=dory_list, data_b=tvm_list,
                                  name_a="Dory", name_b="TVM",
                                  y_title="Cycles",
                                  x_categories=["O0","O1","O2","O3"], x_title="GCC Optimization Level",
                                  barmode="relative",
                                  fig_title="TVM vs Dory - Backend comparison (Absolute Cycles) - GCC Optimization",
                                  legend_title="Backend",
                                  percentage=False))
    figures.append(bar_plot_total_wrapper(data_a=dory_rel_list, data_b=tvm_rel_list,
                                  name_a="Dory", name_b="TVM",
                                  y_title="Cycles (%)",
                                  x_categories=["O0","O1","O2","O3"], x_title="GCC Optimization Level",
                                  barmode="relative",
                                  fig_title="TVM vs Dory - Backend comparison (Relative Cycles) - GCC Optimization",
                                  legend_title="Backend",
                                  percentage=True))
    figures.append(bar_plot_total_wrapper(data_a=cycle_counts_global, 
                                  data_b=tot_list,
                                  name_a="Globally Measured", name_b="Total of individual Kernels",
                                  y_title="Cycles",
                                  x_categories=["O0","O1","O2","O3"], x_title="GCC Optimization Level",
                                  barmode="group",
                                  fig_title="Runtime Overhead Measurement",
                                  legend_title="Measurement",
                                  percentage=False))
    glob_fig = bar_plot_total_wrapper(data_a=cycle_counts_global, 
                                  data_b=None,
                                  name_a="Globally Measured", name_b=None,
                                  y_title="Cycles",
                                  x_categories=["O0","O1","O2","O3"], x_title="GCC Optimization Level",
                                  barmode="group",
                                  fig_title="Global Performance Improvement Chart - GCC Optimization",
                                  legend_title="Measurement", legend=False,
                                  percentage=False)
    annotate_speedup(glob_fig, cycle_counts_global, ["O0","O1","O2","O3"])
    figures.append(glob_fig)
    merge_figures(figures, path=plot_path)
    
