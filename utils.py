# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:47:02 2021

@author: Yang
"""
##############################################################################
# helper functions
##############################################################################

import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from scipy import stats

# generate directory
def gen_dir(hyperparamters):
    directory = 'if_normalize'
    if hyperparamters.if_normalize:
        directory += '_True'
    else:
        directory += '_False'
    directory += '_layers_'
    for i in range(1, len(hyperparamters.layers)):
        directory += (str(hyperparamters.layers[i]) + '_')
    directory += 'concatenate_'
    if hyperparamters.concatenate:
        directory += 'True_'
    else:
        directory += 'False_'
    directory += 'pooling_'    
    directory += hyperparamters.global_pooling
    return directory

# preprocess function
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) 
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized.todense()

# weight initialization function
def weight_variable_glorot(input_dim, output_dim):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010) initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return initial

# define GCN-based encoder and decoder
def GCN_encoder_decoder_classifier(A,F,params,num_labels):
    # ref: https://stackoverflow.com/questions/38235555/tensorflow-matmul-of-input-matrix-with-batch-data
    # each GCN layer is Z = Relu(AFW)
    # A is the preprocessed adj matrix, input denoted as A
    # F is the feature matrix, in this case the 68x68 identity matrix
    # W is the weight matrix to learn 
    num_nodes = int(F.shape[1])
    layers = params[0]
    pooling_type = params[1][0]
    concatenate = params[1][1]
    input_dim, output_dim = layers[0], layers[-1]
    num_layers = len(layers) - 1
    
    layer_output = []
    
    for layer in range(1, len(layers)):
        input_dim = layers[layer-1]
        output_dim = layers[layer]
        W = tf.Variable(weight_variable_glorot(input_dim, output_dim))    
        FW = tf.reshape(tf.reshape(F, [-1, input_dim]) @ W, [-1, num_nodes, output_dim])
        z_layer = tf.matmul(A, FW)   
        z_layer = tf.nn.relu(z_layer)
        layer_output.append(z_layer)
        F = z_layer
    
    node_emb = layer_output[-1]
    if num_layers > 1 and concatenate:
        node_emb = layer_output[0]
        for idx in range(1, len(layer_output)):
            node_emb = tf.concat([node_emb, layer_output[idx]], -1)
        
    # the output of this function should be a matrix representing the recovered FC
    zt = tf.matrix_transpose(node_emb)
    Z = tf.matmul(node_emb, zt)
    outputs = tf.nn.relu(Z)
    #outputs = tf.nn.tanh(outputs) # switch activation functions    
    
    N = int(node_emb.shape[2])
    enc = node_emb
    encc = tf.transpose(enc,[0,2,1])
    if pooling_type == 'ave': 
        Wa = tf.constant(1/num_nodes, shape = [num_nodes,1])  # taking the average of all node embeddings 
        graph_emb = tf.reshape(tf.reshape(encc, [-1, num_nodes]) @ Wa, [-1, N, 1])
    elif pooling_type == 'sum':
        Wa = tf.constant(1/1.0, shape = [num_nodes,1])  # taking the sum of all node embeddings 
        graph_emb = tf.reshape(tf.reshape(encc, [-1, num_nodes]) @ Wa, [-1, N, 1])
    else:
        graph_emb = tf.reduce_max(enc, axis = 1, keepdims = True)
        graph_emb = tf.transpose(graph_emb,[0,2,1])
    
    # now each subject has a feature vector, need to design the logistic regression classifier
    Wb = tf.Variable(tf.truncated_normal([N, num_labels]))
    bia = tf.Variable(tf.zeros([num_labels]))
    xx = tf.transpose(graph_emb, [0,2,1])
    outputs2 = tf.reshape(tf.reshape(xx, [-1, N]) @ Wb, [-1, 1])
    outputs2 = tf.nn.bias_add(outputs2, bia)
    
    return outputs, node_emb, outputs2, graph_emb
    
def plot_results(file_search, L):
    this_res = []
    for item in L:
        if file_search in item[0]:
            if 'sum' in item[0]:
                if 'concatenate_True' in item[0]:
                    for it in item[1]:
                        this_res.append([it, 'con', 'sum'])
                else:
                    for it in item[1]:
                        this_res.append([it, 'no_con', 'sum'])
            elif 'ave' in item[0]:
                if 'concatenate_True' in item[0]:
                    for it in item[1]:
                        this_res.append([it, 'con', 'ave'])
                else:
                    for it in item[1]:
                        this_res.append([it, 'no_con', 'ave'])
            else:
                if 'concatenate_True' in item[0]:
                    for it in item[1]:
                        this_res.append([it, 'con', 'max'])
                else:
                    for it in item[1]:
                        this_res.append([it, 'no_con', 'max'])   
    return this_res    

def t_test(x, y, alternative = 'both-sided'):
    tval, double_p = stats.ttest_ind(x, y, equal_var = False)
    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if tval >= 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if tval < 0:
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return pval