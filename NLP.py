#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:36:40 2020

@author: jay
"""


import tensorflow as tf
import numpy as np

def load_glove(glove_path):
    """
    converts a text file of pretrained glove embeddings into a dict lookup table

    Parameters
    ----------
    glove_path : str
        path to the location of the .txt file containing the glove embedding.

    Returns
    -------
    glove_lookup : dict
        dictionary of pair (word, embed_vector)

    """
    glove_lookup = {};
    with open(glove_path) as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            glove_lookup[word] = coefs;
    
    return glove_lookup


def tokenize_on_ds(tokenizer,ds):   
    """
    fits a tf.keras tokenizer on a tf.dataset of encoded (bytes) strings. note that the ds CANNOT be batched

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.tokenizer
    
    ds : tf.Dataset
        each entry is a text string.

    Yields
    ------
    tokenizer: tf.keras.preprocessing.text.tokenizer
        fitted tokenizer.

    """
    
    def ds_gen(dataset):
        for i in dataset:
            yield i[1].numpy().decode()
    
    gen=ds_gen(ds)
    tokenizer.fit_on_texts(gen)
    
    return tokenizer

def make_embedding_matrix(tokenizer,embedding_lookup,embedding_dim=100):
    """
    prepares a numpy array to be used as weights for a tf.keras Embed layer downstream

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
        
    embedding_lookup : dict
        dictionary of pair (word, embed_vector), pretrained
    embedding_dim : TYPE, optional
        DESCRIPTION. length of the pretrained embed_vectors from embedding_lookup. The default is 100 for GloVE

    Returns
    -------
    embeddings_matrix : ndarray
        numpy array of shape (len(tokenizer.word_index)+1,embedding_dim), where each row is a vector embedding of a word whose
        tokenized integer encoding corresponds to the row number.

    """
    word_index=tokenizer.word_index
    vocab_size=len(word_index)
    
    embeddings_matrix=np.zeros((vocab_size+1,embedding_dim))
    
    for word,encoding in word_index.items():
        embedding_vector=embedding_lookup.get(word)
        if embedding_vector is not None:
            embeddings_matrix[encoding]=embedding_vector
        
    return embeddings_matrix
    
    
    