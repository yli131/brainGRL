# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 12:51:02 2020

@author: Yang
"""

# functions to plot confusion matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import scipy.stats as stats
import scipy.stats as st
from scipy.fftpack import fft, fftfreq 
from scipy.signal import argrelextrema
import operator
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = 'DejaVu Sans'

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Generic function to run any model specified
def perform_model(model, X_train, y_train, X_test, y_test, class_labels, cm_normalize = True, \
                 print_cm = True, cm_cmap = plt.cm.Greens):   
    # to store results at various phases
    results = dict()
    
    # time at which model starts training 
    train_start_time = datetime.now()
    print('training the model..')
    model.fit(X_train, y_train)
    print('Done \n \n')
    train_end_time = datetime.now()
    results['training_time'] =  train_end_time - train_start_time
    print('training_time(HH:MM:SS.ms) - {}\n\n'.format(results['training_time']))
    
    # predict test data
    print('Predicting test data')
    test_start_time = datetime.now()
    y_pred = model.predict(X_test)
    test_end_time = datetime.now()
    print('Done \n \n')
    results['testing_time'] = test_end_time - test_start_time
    print('testing time(HH:MM:SS:ms) - {}\n\n'.format(results['testing_time']))
    results['predicted'] = y_pred

    # calculate overall accuracty of the model
    accuracy = metrics.accuracy_score(y_true = y_test, y_pred = y_pred)
    # store accuracy in results
    results['accuracy'] = accuracy
    print('---------------------')
    print('|      Accuracy      |')
    print('---------------------')
    print('\n    {}\n\n'.format(accuracy))
    
    # confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    results['confusion_matrix'] = cm
    if print_cm: 
        print('--------------------')
        print('| Confusion Matrix |')
        print('--------------------')
        print('\n {}'.format(cm))
        
    # plot confusin matrix
    plt.figure(figsize=(8,8))
    plt.grid(b = False)
    plot_confusion_matrix(cm, classes = class_labels, normalize = True, title = 'Normalized confusion matrix', cmap = cm_cmap)
    plt.show()
    
    # get classification report
    print('-------------------------')
    print('| Classification Report |')
    print('-------------------------')
    classification_report = metrics.classification_report(y_test, y_pred)
    # store report in results
    results['classification_report'] = classification_report
    print(classification_report)
    
    # add the trained  model to the results
    results['model'] = model
    
    return results

# Method to print grid search attributes
def print_grid_search_attributes(model):
    # Estimator that gave highest score among all the estimators formed in GridSearch
    print('--------------------------')
    print('|      Best Estimator     |')
    print('--------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))


    # parameters that gave best results while performing grid search
    print('--------------------------')
    print('|     Best parameters     |')
    print('--------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))


    #  number of cross validation splits
    print('---------------------------------')
    print('|   No of CrossValidation sets   |')
    print('--------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))


    # Average cross validated score of the best estimator, from the Grid Search 
    print('--------------------------')
    print('|        Best Score       |')
    print('--------------------------')
    print('\n\tAverage Cross Validate scores of best estimator : \n\n\t{}\n'.format(model.best_score_))
    
def plot_learningCurve(history, epochs, path):
    epoch_range = range(1,epochs+1)
    plt.plot(epoch_range, history.history['acc'])
    plt.plot(epoch_range, history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper left')
    plt.show()
    
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'], loc = 'upper left')
    plt.savefig(path, dpi = 300)
    plt.show()
    
# standardize data
def scale_data(trainX, testX, standardize):
	# remove overlap
	cut = int(trainX.shape[1] / 2)
	longX = trainX[:, -cut:, :]
	# flatten windows
	longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
	# flatten train and test
	flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
	flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
	# standardize
	if standardize:
		s = StandardScaler()
		# fit on training data
		s.fit(longX)
		# apply to training and test data
		longX = s.transform(longX)
		flatTrainX = s.transform(flatTrainX)
		flatTestX = s.transform(flatTestX)
	# reshape
	flatTrainX = flatTrainX.reshape((trainX.shape))
	flatTestX = flatTestX.reshape((testX.shape))
	return flatTrainX, flatTestX

def get_frames(df, frame_size, hop_size):
    #N_FEATURES = 3
    frames = []
    labels = []
    measurementID = []
    for i in range(0, len(df) - frame_size, hop_size):
        #t = df['Timestamp'].values[i : i + frame_size]
        x = df['X'].values[i : i + frame_size].reshape(frame_size,1)
        y = df['Y'].values[i : i + frame_size].reshape(frame_size,1)
        z = df['Z'].values[i : i + frame_size].reshape(frame_size,1)
        # retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        measureID = stats.mode(df['mID'][i: i + frame_size])[0][0]
        temp = np.concatenate((x,y,z),axis = 1)
        frames.append(np.asarray(temp))
        labels.append(label)
        measurementID.append(measureID)
    # bring the segment into better shape
    #frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    measurementID = np.asarray(measurementID)
    return frames, labels, measurementID

def get_test_frames(df, frame_size, hop_size):
    #N_FEATURES = 3
    frames = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['X'].values[i : i + frame_size].reshape(frame_size,1)
        y = df['Y'].values[i : i + frame_size].reshape(frame_size,1)
        z = df['Z'].values[i : i + frame_size].reshape(frame_size,1)
        temp = np.concatenate((x,y,z),axis = 1)
        frames.append(np.asarray(temp))
    return frames

def stat_area_features(x, Te=1.0):

    mean_ts = np.mean(x, axis=1).reshape(-1,1) # mean
    max_ts = np.amax(x, axis=1).reshape(-1,1) # max
    min_ts = np.amin(x, axis=1).reshape(-1,1) # min
    std_ts = np.std(x, axis=1).reshape(-1,1) # std
    skew_ts = st.skew(x, axis=1).reshape(-1,1) # skew
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1,1) # kurtosis 
    iqr_ts = st.iqr(x, axis=1).reshape(-1,1) # interquartile range
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1,1)),
                               axis=1), axis=1).reshape(-1,1) # median absolute deviation
    area_ts = np.trapz(x, axis=1, dx=Te).reshape(-1,1) # area under curve
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=Te).reshape(-1,1) # area under curve ** 2

    return np.concatenate((mean_ts,max_ts,min_ts,std_ts,skew_ts,kurtosis_ts,
                           iqr_ts,mad_ts,area_ts,sq_area_ts), axis=1)

def frequency_domain_features(x, Te=1.0):

    # As the DFT coefficients and their corresponding frequencies are symetrical arrays
    # with respect to the middle of the array we need to know if the number of readings 
    # in x is even or odd to then split the arrays...
    if x.shape[1]%2 == 0:
        N = int(x.shape[1]/2)
    else:
        N = int(x.shape[1]/2) - 1
    xf = np.repeat(fftfreq(x.shape[1],d=Te)[:N].reshape(1,-1), x.shape[0], axis=0) # frequencies
    dft = np.abs(fft(x, axis=1))[:,:N] # DFT coefficients   
    
    # statistical and area features
    dft_features = stat_area_features(dft, Te=1.0)
    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1,1)
    # 5 first DFT coefficients 
    dft_first_coef = dft[:,:5]    
    # 5 first local maxima of DFT coefficients and their corresponding frequencies
    dft_max_coef = np.zeros((x.shape[0],5))
    dft_max_coef_f = np.zeros((x.shape[0],5))
    for row in range(x.shape[0]):
        # finds all local maximas indexes
        extrema_ind = argrelextrema(dft[row,:], np.greater, axis=0) 
        # makes a list of tuples (DFT_i, f_i) of all the local maxima
        # and keeps the 5 biggest...
        extrema_row = sorted([(dft[row,:][j],xf[row,j]) for j in extrema_ind[0]],
                             key=operator.itemgetter(0), reverse=True)[:5] 
        for i, ext in enumerate(extrema_row):
            dft_max_coef[row,i] = ext[0]
            dft_max_coef_f[row,i] = ext[1]    
    
    return np.concatenate((dft_features,dft_weighted_mean_f,dft_first_coef,
                           dft_max_coef,dft_max_coef_f), axis=1)

def make_feature_vector(x,y,z, Te=1.0):

    # Raw signals :  stat and area features
    features_xt = stat_area_features(x, Te=Te)
    features_yt = stat_area_features(y, Te=Te)
    features_zt = stat_area_features(z, Te=Te)
    '''
    # Jerk signals :  stat and area features
    features_xt_jerk = stat_area_features((x[:,1:]-x[:,:-1])/Te, Te=Te)
    features_yt_jerk = stat_area_features((y[:,1:]-y[:,:-1])/Te, Te=Te)
    features_zt_jerk = stat_area_features((z[:,1:]-z[:,:-1])/Te, Te=Te) 
    '''
    # Raw signals : frequency domain features 
    features_xf = frequency_domain_features(x, Te=1/Te)
    features_yf = frequency_domain_features(y, Te=1/Te)
    features_zf = frequency_domain_features(z, Te=1/Te)
    '''
    # Jerk signals : frequency domain features 
    features_xf_jerk = frequency_domain_features((x[:,1:]-x[:,:-1])/Te, Te=1/Te)
    features_yf_jerk = frequency_domain_features((y[:,1:]-y[:,:-1])/Te, Te=1/Te)
    features_zf_jerk = frequency_domain_features((z[:,1:]-z[:,:-1])/Te, Te=1/Te)
    '''
    # Raw signals correlation coefficient between axis
    cor = np.empty((x.shape[0],3))
    for row in range(x.shape[0]):
        xyz_matrix = np.concatenate((x[row,:].reshape(1,-1),y[row,:].reshape(1,-1),
                                     z[row,:].reshape(1,-1)), axis=0)
        cor[row,0] = np.corrcoef(xyz_matrix)[0,1]
        cor[row,1] = np.corrcoef(xyz_matrix)[0,2]
        cor[row,2] = np.corrcoef(xyz_matrix)[1,2]
    
    return np.concatenate((features_xt, features_yt, features_zt,                           
                           features_xf, features_yf, features_zf,
                           cor), axis=1)    
    
def perform_tsne(X_data, y_data, perplexities, n_iter, directory):       
    # perform t-sne
    perplexity = perplexities
    print('\nperforming tsne with perplexity {} and with {} iterations at max'.format(perplexity, n_iter))
    X_reduced = TSNE(verbose = 2, perplexity = perplexity).fit_transform(X_data)
    print('Done..')
    
    # prepare the data for seaborn         
    print('Creating plot for this t-sne visualization..')
    df = pd.DataFrame({'x':X_reduced[:,0], 'y':X_reduced[:,1] ,'label':y_data})
    
    # draw the plot in appropriate place in the grid
    sns.lmplot(data = df, x = 'x', y = 'y', hue = 'label', fit_reg = False, size = 8,\
               palette = "Set1",markers = ['^','v','s','o', '1'])
    plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
    #img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
    print('saving this plot as image in present working directory...')
    plt.savefig(directory, dpi = 300)
    plt.show()
    print('Done')    
    
def perform_pca(X_data, y_data, n_comp, directory):       
    # perform pca
    pca = PCA(n_components = n_comp)
    principalComponents = pca.fit_transform(X_data)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2']) 
    principalDf['label'] = y_data
    sns.lmplot(data = principalDf, x = 'principal component 1', y = 'principal component 2', hue = 'label', fit_reg = False, size = 8,\
               palette = "Set1", markers = ['^','v','s','o', '1'])
    plt.title('2 component PCA')
    #img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
    print('saving this plot as image in present working directory...')
    plt.savefig(directory, dpi = 300)
    plt.show()
    print('Done')    
    
    
    
    
    
    
    
    
    
    
    
