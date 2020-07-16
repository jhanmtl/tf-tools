#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:36:40 2020

@author: jay
"""


import tensorflow as tf
import numpy as np

def load_glove(glove_path):
    glove_lookup = {};
    with open(glove_path) as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            glove_lookup[word] = coefs;
    
    return glove_lookup

