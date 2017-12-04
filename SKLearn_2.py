# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:35:50 2017

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