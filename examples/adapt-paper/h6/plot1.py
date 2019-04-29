#!/usr/bin/env python
import argparse,re,math,os
import pickle
import copy

import csv

import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import matplotlib.cm as mplcm
import matplotlib.colors as colors

import seaborn as sns

from cycler import cycler


import pickle

from matplotlib import rc
from scipy import interpolate
    
    
    
## Use LaTeX text
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
### for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

#
#   Setup input arguments
parser = argparse.ArgumentParser(description='Take a csv file and create a PES')
parser.add_argument('files', nargs='+', help='List of csv files to process')
parser.add_argument('-z', '--zero',
                    type=float,
                    default=0, 
                    help='Set this value as zero for all data series',  
                    required=False)
parser.add_argument('-title_y', 
                    type=str,
                    help='Title of Y axis',  
                    required=True)
parser.add_argument('-xmin', 
                    type=float,
                    default=argparse.SUPPRESS, 
                    help='Minimim x axis plotted. Default=all data',  
                    required=False)
parser.add_argument('-xmx',
                    type=float,
                    default=argparse.SUPPRESS, 
                    help='Maximum x axis plotted. Default=all data',  
                    required=False)
parser.add_argument('-ymin',
                    type=float,
                    default=argparse.SUPPRESS, 
                    help='Minimim y axis plotted. Default=all data',  
                    required=False)
parser.add_argument('-ymax',
                    type=float,
                    default=argparse.SUPPRESS, 
                    help='Maximum y axis plotted. Default=all data',  
                    required=False)
parser.add_argument('-colors',
                    default=argparse.SUPPRESS, 
                    nargs='+',
                    help='List of colors to use ',  
                    required=False)
parser.add_argument('-scheme',
                    default=argparse.SUPPRESS, 
                    type=str,
                    help='Defined set of options for specific tasks',  
                    choices=['hp_n2_pes1','hp_fh_pes1'],
                    required=False)
parser.add_argument('-skip_legend',
                    default=False, 
                    action="store_true",
                    help='Include legend?',  
                    required=False)
#Probably want to remove these:
parser.add_argument('-c', '--color_scheme',
                    type=str,
                    default='Spectral', 
                    help='Color scheme to use for plotting curves. Default=Spectral',  
                    choices=['jet','spectral','Spectral','cool','winter','gist_rainbow','accent'],
                    required=False)
parser.add_argument('-n_colors',
                    type=int,
                    default=argparse.SUPPRESS, 
                    help='Specify number of colors. Useful for matching colors across plots with different numbers of series',  
                    required=False)
#########
args = vars(parser.parse_args())
files = args['files']

print(files)

#read data
for filename in files:
    series_labels = []
    data = dict()
    x_series  = ""   # This will be used as the x-axis

    with open(filename, 'rU') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        
        headers = []
        for ri in next(csv_reader):
            headers.append(ri)
            series_labels.append(ri)
            data[ri] = []
       
        for row in csv_reader:
            for ri in range(len(row)):
                header = headers[ri]
                if len(row[ri]) > 0:
                    data[header].append(float(row[ri]))
    
    
    
    print(" %-10s %s" %("#lines", "series"))
    for key in data:
        print(" %-10i %s" %(len(data[key]),key))



    y_range = .1
    
    ymin =  - y_range/2
    ymax =  + y_range/2
    
    plt.figure(figsize=(3.5,4.0))
   

    ax = plt.axes()
    #ax.set_ylim( bottom  = ymin)
    #ax.set_ylim( top     = ymax)
    #ax.set_xlim( left  = 0)
    #ax.set_xlim( right = n_samples)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    color_stepsize = (1-.1)/(len(series_labels)-1 )
    alpha_val = .1
    for si in range(1,len(series_labels)):
        alpha_val += color_stepsize 
        print(alpha_val)
        series = series_labels[si]
        print(series)
        series_x = data[series_labels[0]][0:len(data[series])]
        series_x = [x for x in series_x]
        series_y = [x for x in data[series]]
        
        #line1 = plt.plot( series_x, data[series], linestyle='-', marker='', label=series, color=colors[0],
        line1 = plt.plot( series_x, series_y, linestyle='-', marker='', color=colors[0], alpha=alpha_val)
        #line1 = plt.plot( series_x, data[series], linestyle='-', marker='', color=colors[0], alpha=alpha_val)
        #line1 = plt.plot( data[series_labels[0]], data[series], linestyle='-', marker='o', color=colors[si])
    

    x_label = series_labels[0]

    #line1 = plt.plot( data[x_label], data['UCCSD'],         linestyle='-', marker='', color=colors[1], label="UCCSD")
    #line1 = plt.plot( data[x_label], data['HF'],            linestyle='-', marker='', color=colors[2], label="HF")
    #line1 = plt.plot( data[x_label], data['STAG(1e-1)'],    linestyle='-', marker='.', color=colors[3], label="STAG(1e-1)")
    #line1 = plt.plot( data[x_label], data['STAG(1e-2)'],    linestyle='-', marker='.', color=colors[4], label="STAG(1e-2)")
    #line1 = plt.plot( data[x_label], data['STAG(1e-3)'],    linestyle='-', marker='.', color=colors[5], label="STAG(1e-3)")
   
    
    
    handles, labels = ax.get_legend_handles_labels()
    #first_legend = plt.legend(handles=["uCCSD","Mean"], loc=1)


    #fig, ax = plt.subplots()
    #ax.legend(handles, labels, loc=4, prop={'size':10})
    #ax.legend(handles, labels, loc="lower center")
    #ax.set_xlabel("Sample")
    #ax.set_ylabel("E, J")

    plt.xlabel(x_label)
    plt.ylabel(args['title_y'])
    plt.yscale('log')


    """
    lim = .1
    binwidth = .001
    bins = np.arange(-lim, lim + binwidth, binwidth)
    
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_histy = [left_h, bottom, 0.2, height]
    axHisty = plt.axes(rect_histy)
    axHisty.hist(n_samples, bins=bins, orientation='horizontal')
    axHisty.set_ylim(axScatter.get_ylim())
    """
    
   
    plt.savefig(filename+".pdf", facecolor='w', edgecolor='w',
                    dpi = 300, # not used for pdf
                    orientation='portrait', papertype=None, format="pdf",
                    transparent=False, bbox_inches='tight')
    plt.show()


