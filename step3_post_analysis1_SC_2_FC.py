# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:44:02 2021

@author: Yang
"""
##############################################################################
# reorganize the result files due to huge file size
##############################################################################

# Definitions and import packages
import numpy as np
from easydict import EasyDict as edict
from utils import *
import pickle
import os

if not os.path.exists('./post_results'):
    os.mkdir('./post_results')
    
##############################################################################
# load prepared data
Feature_mat = np.load('data/Node_features.npy')
input_dim = Feature_mat.shape[2] # length of graph signals/node features

##############################################################################
# load obtained results
filenames = ['MSE_results', 'CLA_results', 'TPR_results', 'TNR_results', 'Precision_results', 'Recall_results', 'Fscore_results',\
             'training_epochs', 'training_time']#, 'node_embeds', 'graph_embeds', 'subject_ID', 'predicted_labels', \
                 #'trainining_curve_per_fold', 'val_curve_per_fold','TPs', 'TNs', 'FPs', 'FNs']
all_res = []
for _ in filenames:
    all_res.append([])

all_layer_setup = [[input_dim,128,64],[input_dim,64,32],[input_dim,32,16], [input_dim,16,8],
                  [input_dim,128,64,32],[input_dim,64,32,16],[input_dim,32,16,8]]
all_single_layer_setup = [[input_dim, 128],[input_dim, 64], [input_dim, 32],[input_dim, 16], [input_dim,8]] # for single layer, concatenate is useless

for layer_setup in all_layer_setup:
    all_hyperparamters = [edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'ave', 
                  'concatenate': True}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'max', 
                  'concatenate': True}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'sum', 
                  'concatenate': True}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'ave', 
                  'concatenate': False}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'max', 
                  'concatenate': False}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'sum', 
                  'concatenate': False})]
    for hyperparamters in all_hyperparamters:
        directory_name = gen_dir(hyperparamters)
        file = open('results/' + directory_name + '.pkl','rb')
        object_file = pickle.load(file)
        for i in range(len(all_res)):
            all_res[i].append([directory_name, object_file[i]])
        file.close()

for layer_setup in all_single_layer_setup:
    all_hyperparamters = [edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'ave', 
                  'concatenate': True}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'max', 
                  'concatenate': True}),
                          
                          edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'sum', 
                  'concatenate': True})]
    for hyperparamters in all_hyperparamters:
        directory_name = gen_dir(hyperparamters)
        file = open('results/' + directory_name + '.pkl','rb')
        object_file = pickle.load(file)
        for i in range(len(all_res)):
            all_res[i].append([directory_name, object_file[i]])
        file.close()    
    
for xi in range(len(all_res)):
    with open('post_results/' + filenames[xi] + '.pkl', 'wb') as f:
        pickle.dump(all_res[xi], f)
        
print('Process completed and saved')