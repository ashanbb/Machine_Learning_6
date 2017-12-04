# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:36:20 2017

@author: abogollagama
"""


# In[1]:

# ensure common functions across Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We use pickle to import the binary objects we need for subsequent modeling.
# See documentation at https://docs.python.org/3/library/pickle.html

import pickle  # used for dumping and loading binary files

# earlier data collection was defined as follows for pickle.
# data = {
#     'train_data': train_data,
#     'train_labels': train_labels,
#     'validation_data': validation_data,
#     'validation_labels': validation_labels,
#     'test_data': test_data,
#     'test_labels': test_labels}

with open('mnist_data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f)

# extract objects from the dictionary object data
train_data = data['train_data']
train_labels = data['train_labels'] 
validation_data = data['validation_data'] 
validation_labels = data['validation_labels'] 
test_data = data['test_data'] 
test_labels = data['test_labels']  
    
# check data from pickle load
print('\ntrain_data object:', type(train_data), train_data.shape)    
print('\ntrain_labels object:', type(train_labels),  train_labels.shape)  
print('\nvalidation_data object:', type(validation_data),  validation_data.shape)  
print('\nvalidation_labels object:', type(validation_labels),  validation_labels.shape)  
print('\ntest_data object:', type(test_data),  test_data.shape)  
print('\ntest_labels object:', type(test_labels),  test_labels.shape)  

print('\ndata input complete')

# In[2]

import numpy as np
import pandas as pd
import time

# user-defined function to convert binary digits to digits 0-9
def label_transform(y_in):
    for i in range(len(y_in)):
        if (y_in[i] == 1): return i

y_train = []    
for j in range(train_labels.shape[0]):
    y_train.append(label_transform(train_labels[j,]))  
y_train = np.asarray(y_train)    

y_validation = []    
for j in range(validation_labels.shape[0]):
    y_validation.append(label_transform(validation_labels[j,]))  
y_validation = np.asarray(y_validation)    

y_test = []    
for j in range(test_labels.shape[0]):
    y_test.append(label_transform(test_labels[j,]))  
y_test = np.asarray(y_test)    
    
# 28x28 matrix of entries converted to vector of 784 entries    
X_train = train_data.reshape(55000, 784)
X_validation = validation_data.reshape(5000, 784)    
X_test = test_data.reshape(10000, 784)    

# check data intended for Scikit Learn input
print('\nX_train object:', type(X_train), X_train.shape)    
print('\ny_train object:', type(y_train),  y_train.shape)  
print('\nX_validation object:', type(X_validation),  X_validation.shape)  
print('\ny_validation object:', type(y_validation),  y_validation.shape)  
print('\nX_test object:', type(X_test),  X_test.shape)  
print('\ny_test object:', type(y_test),  y_test.shape)    

# In[3] 

# Scikit Learn MLP Classification does validation internally, 
# so there is with no need for a separate validation set.
# We will combine the train and validation sets.

X_train_expanded = np.vstack((X_train, X_validation))
y_train_expanded = np.vstack((y_train.reshape(55000,1), y_validation.reshape(5000,1)))

print('\nX_train_expanded object:', type(X_train_expanded),  X_train_expanded.shape)  
print('\ny_train_expanded object:', type(y_train_expanded), y_train_expanded.shape)    

# In[4]

RANDOM_SEED = 9999

from sklearn.neural_network import MLPClassifier

names = ['ANN-2-Layers-10-Nodes-per-Layer',
         'ANN-2-Layers-20-Nodes-per-Layer',
         'ANN-5-Layers-10-Nodes-per-Layer',
         'ANN-5-Layers-20-Nodes-per-Layer']

layers = [2, 2, 5, 5]
nodes_per_layer = [10, 20, 10, 20]
treatment_condition = [(10, 10), 
                       (20, 20), 
                       (10, 10, 10, 10, 10), 
                       (20, 20, 20, 20, 20)] 

# note that validation is included in the method  
# for validation_fraction 0.083333, note that 60000 * 0.83333 = 5000    
methods = [MLPClassifier(hidden_layer_sizes=treatment_condition[0], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', 
              learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, 
              random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
              nesterovs_momentum=True, early_stopping=False, 
              validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[1], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', 
              learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, 
              random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
              nesterovs_momentum=True, early_stopping=False, 
              validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[2], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', 
              learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, 
              random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
              nesterovs_momentum=True, early_stopping=False, 
              validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    MLPClassifier(hidden_layer_sizes=treatment_condition[3], activation='relu', 
              solver='adam', alpha=0.0001, batch_size='auto', 
              learning_rate='constant', learning_rate_init=0.001, 
              power_t=0.5, max_iter=200, shuffle=True, 
              random_state=RANDOM_SEED, 
              tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
              nesterovs_momentum=True, early_stopping=False, 
              validation_fraction=0.083333, beta_1=0.9, beta_2=0.999, epsilon=1e-08)]
 
index_for_method = 0 
training_performance_results = []
test_performance_results = []
processing_time = []
   
for name, method in zip(names, methods):
    print('\n------------------------------------')
    print('\nMethod:', name)
    print('\n  Specification of method:', method)
    start_time = time.clock()
    method.fit(X_train, y_train)
    end_time = time.clock()
    runtime = end_time - start_time  # seconds of wall-clock time 
    print("\nProcessing time (seconds): %f" % runtime)        
    processing_time.append(runtime)

    # mean accuracy of prediction in training set
    training_performance = method.score(X_train_expanded, y_train_expanded)
    print("\nTraining set accuracy: %f" % training_performance)
    training_performance_results.append(training_performance)

    # mean accuracy of prediction in test set
    test_performance = method.score(X_test, y_test)
    print("\nTest set accuracy: %f" % test_performance)
    test_performance_results.append(test_performance)
                
    index_for_method += 1

# aggregate the results for final report
# using OrderedDict to preserve the order of variables in DataFrame    
from collections import OrderedDict  

results = pd.DataFrame(OrderedDict([('Method Name', names),
                        ('Layers', layers),
                        ('Nodes per Layer', nodes_per_layer),
                        ('Processing Time', processing_time),
                        ('Training Set Accuracy', training_performance_results),
                        ('Test Set Accuracy', test_performance_results)]))

print('\nBenchmark Experiment: Scikit Learn Artificial Neural Networks\n')
print(results)    