# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 14:37:02 2021

@author: Yang
"""

##############################################################################
# prepare graph data (FC, SC, one-hot encoding signals, labels etc.)
##############################################################################

import pandas as pd 
import numpy as np
from scipy.io import loadmat

# load FC and SC data from csv files
num_subjects = 1058
num_brain_regions = 68

SC_data = np.zeros((num_subjects, num_brain_regions, num_brain_regions))
FC_data = np.zeros((num_subjects, num_brain_regions, num_brain_regions))
Feature_mat = np.zeros((num_subjects, num_brain_regions, num_brain_regions))
sub_ID = loadmat('data/sub_ID_1058.mat')['sub_ID']

for i in range(num_subjects):
    print('Processing subject no.' + str(i+1))
    # import and normalize SC data 
    adj_SC = pd.read_csv("data/SC1058/sub" + str(i+1) + "_SC.csv", header = None).values
    SC_data[i] = (adj_SC - np.min(adj_SC)) / (np.max(adj_SC) - np.min(adj_SC))
    # import FC data and remove negative connections
    adj_FC = pd.read_csv("data/FC1058/sub" + str(i+1) + "_FC.csv", header = None).values
    FC_data[i] = np.maximum(adj_FC,0)
    # temporarily use one-hot encoding as node attributes
    Feature_mat[i] = np.identity(num_brain_regions)

# now only focus on these subjects including non-drinker (type 1) and heavy drinker (type 6 and 7)
type1_id = np.where(sub_ID == 1)[0].reshape((1, -1))
type6_id = np.where(sub_ID == 6)[0].reshape((1, -1))
type7_id = np.where(sub_ID == 7)[0].reshape((1, -1))
type2_id = np.concatenate([type6_id, type7_id], axis = 1) # indices of type 1 subjects
num_type0 = type1_id.shape[1] # type0 for label 1
num_type1 = type2_id.shape[1] # type1 for label 6 & 7
print('Non drinker = ' + str(num_type0))
print('Drinker = ' + str(num_type1))

FC_data0 = FC_data[type1_id[0,:],:,:]
SC_data0 = SC_data[type1_id[0,:],:,:]
Feature_mat0 = Feature_mat[type1_id[0,:],:,:]
FC_data1 = FC_data[type2_id[0,:],:,:]
SC_data1 = SC_data[type2_id[0,:],:,:]
Feature_mat1 = Feature_mat[type2_id[0,:],:,:]

FC_data = np.concatenate([FC_data0, FC_data1], axis = 0)
SC_data = np.concatenate([SC_data0, SC_data1], axis = 0)
Feature_mat = np.concatenate([Feature_mat0, Feature_mat1], axis = 0)
sub_ID = np.concatenate([type1_id, type2_id], axis = 1)
sub_type = np.zeros((num_type0 + num_type1, 1)) 
# 0 for non-drinker, 1 for heavy drinker
for i in range(num_type1):
    sub_type[i+num_type0,0] = 1
    
np.save('data/FC_data.npy', FC_data)
np.save('data/SC_data.npy', SC_data)
np.save('data/Node_features.npy', Feature_mat)
np.save('data/Subject_labels.npy', sub_type)
print('Data saved')