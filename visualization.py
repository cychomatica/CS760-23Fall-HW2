import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Callable

def draw_decision_boundary(model_function:Callable, 
                           x1_grid_min:float=1.0, 
                           x1_grid_max:float=-1.0, 
                           x1_num:int=100,
                           x2_grid_min:float=1.0, 
                           x2_grid_max:float=-1.0, 
                           x2_num:int=100,
                           title='Model Decision Boundary Example', 
                           savefile:str=None):
    """`model_function` should be your model's formula for evaluating your decision tree, returning either `0` or `1`.
    \n`grid_abs_bound` represents the generated grids absolute value over the x-axis, default value generates 50 x 50 grid.
    \nUse `grid_abs_bound = 1.0` for question 6 and `grid_abs_bound = 1.5` for question 7.
    \nSet `savefile = 'plot-save-name.png'` to save the resulting plot, adjust colors and scale as needed."""

    colors=['#91678f','#afd6d2'] # hex color for [y=0, y=1]

    # xval = np.linspace(grid_abs_bound,-grid_abs_bound,int(grid_abs_bound) * 10).tolist() # grid generation
    x1val = np.linspace(x1_grid_min, x1_grid_max, x1_num).tolist() 
    x2val = np.linspace(x2_grid_min, x2_grid_max, x2_num).tolist()
    xdata = []
    for i in range(len(x1val)):
        for j in range(len(x2val)):
            xdata.append([x1val[i],x2val[j]])

    df = pd.DataFrame(data=xdata,columns=['x_1','x_2']) # creates a dataframe to standardize labels
    df['y'] = df.apply(model_function,axis=1) # applies model from model_function arg
    d_columns = df.columns.to_list() # grabs column headers
    y_label = d_columns[-1] # uses last header as label
    d_xfeature = d_columns[0] # uses first header as x_1 feature
    d_yfeature = d_columns[1] # uses second header as x_1 feature
    df = df.sort_values(by=y_label) # sorts by label to ensure correct ordering in plotting loop

    d_xlabel = f"feature $\mathit{{{d_xfeature}}}$" # label for x-axis
    dy_ylabel = f"feature $\mathit{{{d_yfeature}}}$" # label for y-axis
    plt.xlabel(d_xlabel, fontsize=10) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10) # set y-axis label
    legend_labels = [] # create container for legend labels to ensure correct ordering

    for i,label in enumerate(df[y_label].unique().tolist()): # loop through placeholder dataframe
        df_set = df[df[y_label]==label] # sort according to label
        set_x = df_set[d_xfeature] # grab x_1 feature set
        set_y = df_set[d_yfeature] # grab x_2 feature set
        plt.scatter(set_x,set_y,c=colors[i],marker='s', s=40) # marker='s' for square, s=40 for size of squares large enough
        legend_labels.append(f"""{y_label} = {label}""") # apply labels for legend in the same order as sorted dataframe

    plt.title(title, fontsize=12) # set plot title
    ax = plt.gca() # grab to set background color of plot
    ax.set_facecolor('#2b2d2e') # set aforementioned background color in hex color
    plt.legend(legend_labels) # create legend with sorted labels

    if savefile is not None: # save your plot as .png file
        plt.savefig(savefile)
    plt.show() # show plot with decision bounds
    plt.close()

def data_scatter_plot(data, title='Data Set', savefile:str=None):

    colors=['#ed3215','#5fd96e'] # hex color for [y=0, y=1]

    df = pd.DataFrame(data=data,columns=['x_1','x_2','y']) # creates a dataframe to standardize labels
    d_columns = df.columns.to_list() # grabs column headers
    y_label = d_columns[-1] # uses last header as label
    d_xfeature = d_columns[0] # uses first header as x_1 feature
    d_yfeature = d_columns[1] # uses second header as x_1 feature
    df = df.sort_values(by=y_label) # sorts by label to ensure correct ordering in plotting loop

    d_xlabel = f"feature $\mathit{{{d_xfeature}}}$" # label for x-axis
    dy_ylabel = f"feature $\mathit{{{d_yfeature}}}$" # label for y-axis
    plt.xlabel(d_xlabel, fontsize=10) # set x-axis label
    plt.ylabel(dy_ylabel, fontsize=10) # set y-axis label
    legend_labels = [] # create container for legend labels to ensure correct ordering

    for i,label in enumerate(df[y_label].unique().tolist()): # loop through placeholder dataframe
        df_set = df[df[y_label]==label] # sort according to label
        set_x = df_set[d_xfeature] # grab x_1 feature set
        set_y = df_set[d_yfeature] # grab x_2 feature set
        plt.scatter(set_x, set_y, c=colors[i], marker=['^', 'o'][i], s=40) # marker='s' for square, s=40 for size of squares large enough
        legend_labels.append(f"""{y_label} = {label}""") # apply labels for legend in the same order as sorted dataframe

    plt.title(title, fontsize=12) # set plot title
    ax = plt.gca() # grab to set background color of plot
    plt.legend(legend_labels) # create legend with sorted labels

    if savefile is not None: # save your plot as .png file
        plt.savefig(savefile)
    plt.show() # show plot with decision bounds
    plt.close()
