# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:32:28 2021

@author: Yang
"""

##############################################################################
# Analysis of results from main script
##############################################################################

import numpy as np
from utils import *
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(style = "whitegrid")
import os
import operator

if not os.path.exists('./figures'):
    os.mkdir('./figures')
    os.mkdir('./figures/mse')
    os.mkdir('./figures/acc')
    os.mkdir('./figures/precision')
    os.mkdir('./figures/recall')
    os.mkdir('./figures/fscore')
    os.mkdir('./figures/training_time')
    os.mkdir('./figures/training_epoch')
    
Feature_mat = np.load('data/Node_features.npy')
input_dim = Feature_mat.shape[2] # length of graph signals/node features
all_layer_setup = [[input_dim,128,64],[input_dim,64,32],[input_dim,32,16], [input_dim,16,8],
                  [input_dim,128,64,32],[input_dim,64,32,16],[input_dim,32,16,8],
                  [input_dim, 128],[input_dim, 64], [input_dim, 32],[input_dim, 16],[input_dim,8]]

##############################################################################
# Analysis 1
# FC reconstruction MSE
file = open('post_results/MSE_results.pkl','rb')
MSE_res = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, MSE_res)
    df = pd.DataFrame(this_res, columns= ['mse','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "mse", hue = "concatenate", data = df, palette = "Set3", showfliers = False).set_title(file_search + ' regression MSE')
    plt.savefig('figures/mse/' + filename + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
##############################################################################
# Analysis 2
# classification acc
file = open('post_results/CLA_results.pkl','rb')
CLA_res = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, CLA_res)
    df = pd.DataFrame(this_res, columns= ['acc','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "acc", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' classification acc')
    ax.set(ylim=(0.4,1))
    plt.savefig('figures/acc/' + filename + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

##############################################################################
# Analysis 3
# classification precision
file = open('post_results/Precision_results.pkl','rb')
precision_res = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, precision_res)
    df = pd.DataFrame(this_res, columns= ['pre','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "pre", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' classification precision')
    ax.set(ylim=(0.4,1))
    plt.savefig('figures/precision/' + filename + '.png', dpi = 300)
    plt.show()

##############################################################################
# Analysis 4
# classification recall
file = open('post_results/Recall_results.pkl','rb')
recall_res = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, recall_res)
    df = pd.DataFrame(this_res, columns= ['rec','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "rec", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' classification recall')
    ax.set(ylim=(0.4,1))
    plt.savefig('figures/recall/' + filename + '.png', dpi = 300)
    plt.show()

##############################################################################
# Analysis 5
# classification fscore
file = open('post_results/Fscore_results.pkl','rb')
fscore_res = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, fscore_res)
    df = pd.DataFrame(this_res, columns= ['fscore','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "fscore", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' classification fscore')
    ax.set(ylim=(0.4,1))
    plt.savefig('figures/fscore/' + filename + '.png', dpi = 300)
    plt.show()

##############################################################################
# Analysis 6
# training time
file = open('post_results/training_time.pkl','rb')
training_time = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, training_time)
    df = pd.DataFrame(this_res, columns= ['training_time','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "training_time", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' training_time')
    #ax.set(ylim=(0,200))
    plt.savefig('figures/training_time/' + filename + '.png', dpi = 300)
    plt.show()

##############################################################################
# Analysis 7
# training epochs
file = open('post_results/training_epochs.pkl','rb')
training_epoch = pickle.load(file)
for x in all_layer_setup:
    xx = []
    for xitem in x[1:]:
        xx.append(str(xitem))
    file_search = 'layers_' + '_'.join(xx) + '_concatenate'
    filename = 'layers_' + '_'.join(xx)
    this_res = plot_results(file_search, training_epoch)
    df = pd.DataFrame(this_res, columns= ['training_epoch','concatenate','pool'])
    ax = sns.boxplot(x = "pool", y = "training_epoch", hue = "concatenate", data = df, palette = "Set3", showfliers = False)
    ax.set_title(file_search + ' training_epoch')
    #ax.set(ylim=(0,200))
    plt.savefig('figures/training_epoch/' + filename + '.png', dpi = 300)
    plt.show()

##############################################################################
# Analysis 8
# find best model
file = open('post_results/MSE_results.pkl','rb')
MSE_res = pickle.load(file)
file = open('post_results/CLA_results.pkl','rb')
CLA_res = pickle.load(file)
file = open('post_results/Fscore_results.pkl','rb')
fscore_res = pickle.load(file)
file = open('post_results/training_time.pkl','rb')
training_time = pickle.load(file)

all_model_candidates = []
for x in MSE_res:
    all_model_candidates.append(x[0])

model_res = {}
for idx in range(len(all_model_candidates)):
    # criteria
    # acc * fscore / (MSE * training_time)
    model_res[all_model_candidates[idx]] = np.mean(CLA_res[idx][1]) * np.mean(fscore_res[idx][1]) / (np.mean(MSE_res[idx][1]) * np.mean(training_time[idx][1]))

sorted_model_res = sorted(model_res.items(), key = operator.itemgetter(1), reverse = True)

num_of_selected = 3
selected_models = []
idx = 0
while len(selected_models) < num_of_selected:
    cur = sorted_model_res[idx]
    if 'False' not in cur[0] and 'ave' in cur[0]:
        selected_models.append(cur[0])
    idx += 1

print(selected_models)   

















