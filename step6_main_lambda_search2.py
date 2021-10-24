# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:44:56 2021

@author: Yang
"""

##############################################################################
# Do an extensive search by testing the lambda values below on the same 10 folds used previously
##############################################################################

# Definitions and import packages
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
from utils import *
import pickle
import pandas as pd
import seaborn as sns
import os

if not os.path.exists('figures/lambda_search/acc/'):
    os.mkdir('figures/lambda_search/')
    os.mkdir('figures/lambda_search/acc/')
    os.mkdir('figures/lambda_search/mse/')
    os.mkdir('figures/lambda_search/fscore/')
    
##############################################################################
# load prepared data
Feature_mat = np.load('data/Node_features.npy')

# fixed parameters
input_dim = Feature_mat.shape[2] # length of graph signals/node features

##############################################################################
# define model hyperparameters
all_layer_setup = [[input_dim,64,32,16],[input_dim,32,16,8],[input_dim,128,64,32]]
model_name = []
for layer_setup in all_layer_setup:
    all_hyperparameters = [edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'ave', 
                  'concatenate': True})]
    
    for hyperparameters in all_hyperparameters:  
        output_dim = hyperparameters.layers[-1]
        # change the output dimension depending on whether concatenating node embeddings or not
        if hyperparameters.concatenate:
            output_dim = sum(hyperparameters.layers[1:])
        args = [hyperparameters.layers,[hyperparameters.global_pooling, hyperparameters.concatenate]]
        
        # predefine directory to save results w.r.t different combos of options
        directory_name = gen_dir(hyperparameters)
        model_name.append(directory_name)
#print(model_name)

lambs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
model_lambda_perf_mse = []
model_lambda_perf_fscore = []
model_lambda_perf_acc = []
for md in model_name:
    curmse = []
    curfs = []
    curacc = []
    for x in lambs:
        file = open('results/lambda_search/' + md + '_lambda_' + str(x) + '.pkl','rb')
        temp_file = pickle.load(file)
        curmse.append(temp_file[0])
        curacc.append(temp_file[1])
        curfs.append(temp_file[6])
        del temp_file
    model_lambda_perf_mse.append(curmse)
    model_lambda_perf_fscore.append(curfs)
    model_lambda_perf_acc.append(curacc)

for idxx in range(len(model_lambda_perf_mse)):
    mx = model_lambda_perf_mse[idxx]
    this_res = []
    for idx in range(len(mx)):
        lam = str(lambs[idx])
        for item in mx[idx]:
            this_res.append([item,lam])
    df = pd.DataFrame(this_res, columns= ['mse','lambda'])
    ax = sns.boxplot(x = "lambda", y = "mse", data = df, showfliers = False).set_title('regression MSE lambda search')
    plt.savefig('figures/lambda_search/mse/' + model_name[idxx] + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

for idxx in range(len(model_lambda_perf_acc)):
    mx = model_lambda_perf_acc[idxx]
    this_res = []
    for idx in range(len(mx)):
        lam = str(lambs[idx])
        for item in mx[idx]:
            this_res.append([item,lam])
    df = pd.DataFrame(this_res, columns= ['acc','lambda'])
    ax = sns.boxplot(x = "lambda", y = "acc", data = df, showfliers = False).set_title('acc lambda search')
    plt.savefig('figures/lambda_search/acc/' + model_name[idxx] + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

for idxx in range(len(model_lambda_perf_fscore)):
    mx = model_lambda_perf_fscore[idxx]
    this_res = []
    for idx in range(len(mx)):
        lam = str(lambs[idx])
        for item in mx[idx]:
            this_res.append([item,lam])
    df = pd.DataFrame(this_res, columns= ['fscore','lambda'])
    ax = sns.boxplot(x = "lambda", y = "fscore", data = df, showfliers = False).set_title('fscore lambda search')
    plt.savefig('figures/lambda_search/fscore/' + model_name[idxx] + '.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

##############################################################################
# find the best model + lambda
ave_model_lambda_perf_acc = []
for x in model_lambda_perf_acc:
    xtmp = []
    for xx in x:
        xtmp.append(np.mean(xx))
    ave_model_lambda_perf_acc.append(xtmp)
ave_model_lambda_perf_fscore = []
for x in model_lambda_perf_fscore:
    xtmp = []
    for xx in x:
        xtmp.append(np.mean(xx))
    ave_model_lambda_perf_fscore.append(xtmp)    
ave_model_lambda_perf_mse = []
for x in model_lambda_perf_mse:
    xtmp = []
    for xx in x:
        xtmp.append(np.mean(xx))
    ave_model_lambda_perf_mse.append(xtmp)

performance_mat = [[0] * len(lambs) for _ in range(3)]
for i in range(len(lambs)):
    for j in range(3):
        performance_mat[j][i] = ave_model_lambda_perf_acc[j][i] * ave_model_lambda_perf_fscore[j][i] / ave_model_lambda_perf_mse[j][i]
idxi, idxj = 0, 0
bestperf = -1
for i in range(len(lambs)):
    for j in range(3):
        if performance_mat[j][i] > bestperf:
            bestperf = performance_mat[j][i]
            idxi = i
            idxj = j
            
print('choose lambda = ' + str(lambs[idxi]))
print('choose model = ' + str(model_name[idxj]))




