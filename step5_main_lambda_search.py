# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:26:41 2021

@author: Yang
"""

##############################################################################
# Do an extensive search by testing the lambda values below on the same 10 folds used previously
##############################################################################

# Definitions and import packages
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict
from utils import *
import timeit
import pickle
import os

if not os.path.exists('./models/lambda_search/'):
    os.mkdir('./models/lambda_search/')
    os.mkdir('./results/lambda_search/')
    
##############################################################################
# load prepared data
FC_data = np.load('data/FC_data.npy')
SC_data_orig = np.load('data/SC_data.npy')
Feature_mat = np.load('data/Node_features.npy')
sub_type = np.load('data/Subject_labels.npy')
SC_data = np.load('data/SC_data.npy')
sub_IDs = np.array(list(range(len(FC_data))))

# fixed parameters
input_dim = Feature_mat.shape[2] # length of graph signals/node features
num_nodes = FC_data.shape[1] # number of nodes
num_labels = sub_type.shape[1] # classification output dimension

##############################################################################
# define model hyperparameters
all_layer_setup = [[input_dim,64,32,16],[input_dim,32,16,8], [input_dim,128,64,32]]
for layer_setup in all_layer_setup:
    all_hyperparameters = [edict({'if_normalize': True, 
                  'layers':layer_setup,
                  'global_pooling': 'ave', 
                  'concatenate': True})]
    
    for hyperparameters in all_hyperparameters:  
        if hyperparameters.if_normalize:
            # follow the renormalization trick
            SC_data = np.zeros(SC_data_orig.shape)
            for gi in range(len(SC_data_orig)):
                SC_data[gi] = preprocess_graph(SC_data_orig[gi])
        
        output_dim = hyperparameters.layers[-1]
        # change the output dimension depending on whether concatenating node embeddings or not
        if hyperparameters.concatenate:
            output_dim = sum(hyperparameters.layers[1:])
        args = [hyperparameters.layers,[hyperparameters.global_pooling, hyperparameters.concatenate]]
        
        # predefine directory to save results w.r.t different combos of options
        directory_name = gen_dir(hyperparameters)
        
        ######################################################################
        # define training hyperparameters
        rate = 0.001  # learning rate
        BATCH_SIZE = 64
        bat_size = 20
        early_stopping_patience = 10 
        EPOCHS = 3000
        # tuning paramter controlling trade-off between reconstruction and classification
        lambs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5] 
        
        for lamb in lambs:    
            
            folds = 10  
            # pre-allocate variables to save results generated during training
            MSE_results = [] # record the MSE reconstruction loss at each fold
            CLA_results = [] # record the classification acc at each fold
            TPR_results = [] # record the classification tpr at each fold
            TNR_results = [] # record the classification tnr at each fold
            Precision_results = [] # record the classification precision at each fold
            Recall_results = [] # record the classification recall at each fold
            Fscore_results = [] # record the classification fscore at each fold
            training_epochs = [] # record the number of epochs run at each fold
            training_time = [] # record the training time at each fold
            node_embeds = [] # saved in the order of training, val, test
            graph_embeds = [] # saved in the order of training, val, test
            subject_ID = [] # saved in the order of training, val, test
            predicted_labels = [] # saved in the order of training, val, test
            trainining_curve_per_fold = [] # total loss, rec_loss, class_loss
            val_curve_per_fold = [] # total loss, rec_loss, class_loss
            TPs, TNs, FPs, FNs = [], [], [], [] # saved in the order of test, training, val
            
            ##################################################################
            # pipeline
            for it in range(folds):
                print("Fold ", it)  
            
                # features and labels
                x = tf.placeholder(tf.float32, (None, num_nodes, num_nodes))
                y = tf.placeholder(tf.float32, (None, num_nodes, num_nodes))
                f = tf.placeholder(tf.float32, (None, num_nodes, input_dim))
                t = tf.placeholder(tf.float32, (None, num_labels)) # each subject has a label indicating his drinking type
                
                # training pipeline
                [rec_logits, encoded, classif_logits, graph_feature_vec] = GCN_encoder_decoder_classifier(x,f,args,num_labels)
                cost1 = tf.reduce_mean(tf.squared_difference(rec_logits, y))  # MSE error, FC reconstruction loss
                cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = t, logits = classif_logits)) # sigmoid classification loss 
                cost = cost1 + (lamb * cost2)  # the cost to minimize to train the network
                
                optimizer = tf.train.AdamOptimizer(learning_rate = rate)
                training_operation = optimizer.minimize(cost)
                                                  
                # model evaluation
                rec_error1 = tf.reduce_mean(tf.squared_difference(rec_logits, y))
                rec_error2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = t, logits = classif_logits))
                rec_error = rec_error1 + (lamb * rec_error2)
                
                saver = tf.train.Saver()
            
                def evaluate(x_data, y_data, f_data, t_data):
                    num_examples = len(x_data)
                    embeddings = np.zeros((num_examples,num_nodes,output_dim)) # node embeddings
                    emb_final = np.zeros((num_examples,output_dim,1)) # graph embeddings
                    sigmoid_res = np.zeros((num_examples,1))
                    total_error = 0    # total loss
                    total_error1 = 0   # reconstruction loss
                    total_error2 = 0   # classification loss
                    
                    sess = tf.get_default_session()
                    for offset in range(0, num_examples, bat_size):
                        end = offset + bat_size
                        batch_x, batch_y, batch_f, batch_t = x_data[offset:offset+bat_size], y_data[offset:offset+bat_size], f_data[offset:offset+bat_size], t_data[offset:offset+bat_size]
                        opt = sess.run([rec_error,encoded,graph_feature_vec,classif_logits, rec_error1, rec_error2], feed_dict={x: batch_x, y: batch_y, f: batch_f, t: batch_t})
                        total_error += (opt[0] * len(batch_x))
                        total_error1 += (opt[4] * len(batch_x))
                        total_error2 += (opt[5] * len(batch_x))
                        embeddings[offset:end] = opt[1]
                        emb_final[offset:end] = opt[2]
                        sigmoid_res[offset:end] = opt[3]
                        
                    return total_error / num_examples, embeddings, emb_final, sigmoid_res, total_error1 / num_examples, total_error2 / num_examples
                
                # split dataset into 80% training, 10% validation and 10% test set, stratified
                x_train, x_test, y_train, y_test, f_train, f_test, sub_type_train, sub_type_test, sub_ID_train, sub_ID_test = train_test_split(SC_data, FC_data, Feature_mat, sub_type, sub_IDs, stratify = sub_type, test_size = 0.2, random_state = it)
                x_test, x_val, y_test, y_val, f_test, f_val, sub_type_test, sub_type_val, sub_ID_val, sub_ID_test = train_test_split(x_test, y_test, f_test, sub_type_test, sub_ID_test, stratify = sub_type_test, test_size = 0.5, random_state = it)
            
                start = timeit.default_timer()
                te, te1, te2 = [], [], []
                ve, ve1, ve2 = [], [], []
                
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    num_examples = len(x_train)
                    
                    embeddings_training = np.zeros((num_examples,num_nodes,output_dim)) # node embedding
                    emb_training = np.zeros((num_examples,output_dim,1)) # graph embedding
                    sigmoid_train = np.zeros((num_examples, 1))
                    train_err = np.zeros(EPOCHS)  # total training loss
                    train_err1 = np.zeros(EPOCHS) # reconstruction training loss
                    train_err2 = np.zeros(EPOCHS) # classification training loss
                    val_err = np.zeros(EPOCHS) # total validation loss
                    val_err1 = np.zeros(EPOCHS)
                    val_err2 = np.zeros(EPOCHS)
                    acc = np.zeros(EPOCHS)
                    print("Training...")
                    print()
                    
                    num_increase = 0   # calculate how many consecutive steps of increasing validation loss for early stopping
                    num_increase2 = 0
                    num_increase1 = 0
                    
                    stop_epoch = 0
                    for i in range(EPOCHS):
                        training_error = 0
                        cost_error1 = 0
                        cost_error2 = 0
                        acc1 = 0
                        
                        for offset in range(0, num_examples, BATCH_SIZE):
                            end = offset + BATCH_SIZE
                            batch_x, batch_y, batch_f, batch_t = x_train[offset:end], y_train[offset:end], f_train[offset:end],sub_type_train[offset:end]
                            outs = sess.run([training_operation,cost, encoded, graph_feature_vec, classif_logits, cost1, cost2], feed_dict={x: batch_x, y: batch_y, f: batch_f, t: batch_t})
                            embeddings_training[offset:end] = outs[2]
                            emb_training[offset:end] = outs[3]
                            sigmoid_train[offset:end] = outs[4]
                            training_error += (outs[1] * len(batch_x))
                            cost_error1 += (outs[5] * len(batch_x))
                            cost_error2 += (outs[6] * len(batch_x))
                            
                            # evaluate classification accuracy
                            ttt = 0
                            v = outs[4]
                            for idx in range(v.shape[0]):
                                temp = 1 / (1 + math.exp(-v[idx,0]))
                                ttt += (int((temp >= 0.5)) == int(batch_t[idx,0]))
                            acc1 += ttt
                
                        train_err[i] = training_error / num_examples
                        train_err1[i] = cost_error1 / num_examples
                        train_err2[i] = cost_error2 / num_examples
                        acc[i] = acc1 / num_examples
                        print("EPOCH {} ...".format(i+1))
                        print("MSE on training set is = {:.7f}".format(cost_error1 / num_examples))  
                        print("CLA loss on training set is = {:.7f}".format(cost_error2 / num_examples))  
                        print("Classification accuracy on training set is = {:.7f}".format(acc[i])) 
                        
                        [validation_error, embeddings_val, emb_val, sigmoid_val, val_rec_error, val_clas_error] = evaluate(x_val, y_val, f_val, sub_type_val)
                        #print("EPOCH {} ...".format(i+1))
                        val_err[i] = validation_error
                        val_err1[i] = val_rec_error
                        val_err2[i] = val_clas_error
                        print("MSE on validation set is = {:.7f}".format(val_rec_error))
                        print("CLA loss on validation set is = {:.7f}".format(val_clas_error))
                    
                        ttt = 0
                        v = sigmoid_val
                        for idx in range(v.shape[0]):
                            temp = 1 / (1 + math.exp(-v[idx,0]))
                            ttt += (int((temp >= 0.5)) == int(sub_type_val[idx,0]))
                        
                        print("Classification accuracy on validation set is: {:.7f}".format(ttt / v.shape[0])) 
                        
                        if i > 0:
                            if validation_error > val_err[i-1]:
                                num_increase += 1
                            else:
                                num_increase = 0
                            '''
                            if val_rec_error > val_err1[i-1]:
                                num_increase1 += 1
                            else:
                                num_increase1 = 0
                            if val_clas_error > val_err2[i-1]:
                                num_increase2 += 1
                            else:
                                num_increase2 = 0
                            '''
                        if early_stopping_patience in [num_increase, num_increase1, num_increase2]:
                            stop_epoch = i + 1     
                            break
                        
                    te.append(train_err)
                    te1.append(train_err1)
                    te2.append(train_err2)
                    ve.append(val_err)
                    ve1.append(val_err1)
                    ve2.append(val_err2)
                    
                    saver.save(sess, './models/lambda_search/' + directory_name + '_gcn_encoder_decoder_system_lambda_' + str(lamb) + '_fold_' + str(it))
                    print("Model saved") 
                
                if stop_epoch == 0:
                    stop_epoch = EPOCHS
                training_epochs.append(stop_epoch)
                stop = timeit.default_timer()
                training_time.append(stop - start)
                trainining_curve_per_fold.append([te,te1,te2]) # total, rec, class
                val_curve_per_fold.append([ve,ve1,ve2]) # total, rec, class
                
                # evaluate on test set
                with tf.Session() as sess:
                    saver.restore(sess, tf.train.latest_checkpoint('./models/lambda_search'))
                
                    [test_error, embeddings_testing, emb_test, sigmoid_test, test_rec_error, test_clas_error] = evaluate(x_test, y_test, f_test, sub_type_test)
                    print("Test loss = {:.7f}".format(test_error))
                    MSE_results.append(test_rec_error)
                    ttt = 0
                    v = sigmoid_test
                    tp = 0
                    tn = 0
                    fp = 0
                    fn = 0
                    for idx in range(v.shape[0]):
                        temp = 1 / (1 + math.exp(-v[idx,0]))
                        ttt += (int((temp >= 0.5)) == int(sub_type_test[idx,0]))
                        if int((temp >= 0.5)) == 1 and int(sub_type_test[idx,0]) == 1:
                            tp += 1
                        elif int((temp >= 0.5)) == 0 and int(sub_type_test[idx,0]) == 0:
                            tn += 1    
                        elif int((temp >= 0.5)) == 0 and int(sub_type_test[idx,0]) == 1:
                            fn += 1    
                        else:
                            fp += 1
                            
                    print("MSE on test set is = {:.7f}".format(test_rec_error))
                    print("CLA loss on test set is = {:.7f}".format(test_clas_error))
                    print("Classification accuracy on test set is: {:.7f}".format(ttt / v.shape[0])) 
                    print("TPR on test set is: {:.7f}".format(tp/np.sum(sub_type_test))) 
                    print("TNR on test set is: {:.7f}".format(tn/(len(sub_type_test) - np.sum(sub_type_test))))
                    print("F score on test set is: {:.7f}".format(2 * tp / (2 * tp + fp + fn)))
                    
                    CLA_results.append(ttt / v.shape[0])
                    TPR_results.append(tp/np.sum(sub_type_test))
                    TNR_results.append(tn/(len(sub_type_test) - np.sum(sub_type_test)))
                    Fscore_results.append(2 * tp / (2 * tp + fp + fn))
                    if tp + fp == 0:
                        Precision_results.append(0)
                    else:
                        Precision_results.append(tp / float((tp + fp)))
                    Recall_results.append(tp / float((tp + fn)))
                    
                    v1 = sigmoid_train
                    tp_train = 0
                    tn_train = 0
                    fp_train = 0
                    fn_train = 0
                    ttt_train = 0
                    for idx in range(v1.shape[0]):
                        temp = 1 / (1 + math.exp(-v1[idx,0]))
                        ttt_train += (int((temp >= 0.5)) == int(sub_type_train[idx,0]))
                        if int((temp >= 0.5)) == 1 and int(sub_type_train[idx,0]) == 1:
                            tp_train += 1
                        elif int((temp >= 0.5)) == 0 and int(sub_type_train[idx,0]) == 0:
                            tn_train += 1    
                        elif int((temp >= 0.5)) == 0 and int(sub_type_train[idx,0]) == 1:
                            fn_train += 1   
                        else:
                            fp_train += 1
                            
                    v2 = sigmoid_val
                    tp_val = 0
                    tn_val = 0
                    fp_val = 0
                    fn_val = 0
                    ttt_val = 0
                    for idx in range(v2.shape[0]):
                        temp = 1 / (1 + math.exp(-v2[idx,0]))
                        ttt_val += (int((temp >= 0.5)) == int(sub_type_val[idx,0]))
                        if int((temp >= 0.5)) == 1 and int(sub_type_val[idx,0]) == 1:
                            tp_val += 1
                        elif int((temp >= 0.5)) == 0 and int(sub_type_val[idx,0]) == 0:
                            tn_val += 1   
                        elif int((temp >= 0.5)) == 0 and int(sub_type_val[idx,0]) == 1:
                            fn_val += 1  
                        else:
                            fp_val += 1
                    
                    TP = tp + tp_train + tp_val
                    TN = tn + tn_train + tn_val
                    FN = fn + fn_train + fn_val
                    FP = fp + fp_train + fp_val
                    tt_acc = ttt + ttt_train + ttt_val
                    print("Classification accuracy on whole dataset is: {:.7f}".format(tt_acc / len(sub_type))) 
                    print("TPR on whole dataset is: {:.7f}".format(TP/np.sum(sub_type))) 
                    print("TNR on whole dataset is: {:.7f}".format(TN/(len(sub_type) - np.sum(sub_type))) )
                    print("F score on whole dataset is: {:.7f}".format(2 * TP / (2 * TP + FP + FN)))
                    
                    all_node_embeddings = [embeddings_training, embeddings_val, embeddings_testing]
                    all_graph_embeddings = [emb_training, emb_val, emb_test]
                    all_subject_IDs = [sub_ID_train, sub_ID_val, sub_ID_test]
                    
                    node_embeds.append(all_node_embeddings)
                    graph_embeds.append(all_graph_embeddings)
                    subject_ID.append(all_subject_IDs)
                    predicted_labels.append([v1, v2, v])
                    TPs.append([tp,tp_train, tp_val]) # test, train, val
                    TNs.append([tn,tn_train, tn_val])
                    FPs.append([fp,fp_train, fp_val])
                    FNs.append([fn,fn_train, fn_val])
                    
                tf.keras.backend.clear_session()
            
            ##################################################################
            # saving all results under the current settings
            all_results = [MSE_results, CLA_results, TPR_results, TNR_results, Precision_results, \
                        Recall_results, Fscore_results, training_epochs, training_time, \
                        node_embeds, graph_embeds, subject_ID, predicted_labels, trainining_curve_per_fold, val_curve_per_fold,\
                        TPs, TNs, FPs, FNs]
            with open('results/lambda_search/' + directory_name + '_lambda_' + str(lamb) + '.pkl', 'wb') as f:
                pickle.dump(all_results, f)

print("Lambda search finished")