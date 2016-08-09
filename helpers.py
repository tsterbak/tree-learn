'''
Created on 09.08.2016

@author: Tobias
'''
from math import log
import numpy as np

def entropy(y):
    possible_labels = set(y)
    ent=0.0
    n = len(y)
    counts = np.bincount(np.array(y))
    for label in possible_labels:
        p = float(counts[label])/n
        ent = ent - p*np.log2(p)
    return ent

def divide_set(samples,targets, feature, value):
    lower_index = samples[:,feature] >= value
    lower_set = samples[lower_index]
    lower_set_target = targets[lower_index]
    
    higher_index = samples[:,feature] < value
    higher_set = samples[higher_index]
    higher_set_target = targets[higher_index]
    return lower_set,lower_set_target, higher_set, higher_set_target