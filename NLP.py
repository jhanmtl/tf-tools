#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:36:40 2020

@author: jay
"""


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_embed_txt(txt_path):
    """
    converts a text file of pretrained glove embeddings into a dict lookup table

    Parameters
    ----------
    txt_path : str
        path to the location of the .txt file containing the glove embedding.

    Returns
    -------
    glove_lookup : dict
        dictionary of pair (word, embed_vector)

    """
    lookup = {};
    with open(txt_path) as f:
        for line in f:
            values = line.split();
            word = values[0];
            coefs = np.asarray(values[1:], dtype='float32');
            lookup[word] = coefs;
    
    return lookup



def get_embedding_weights(tokenizer,embedding_lookup):
    """
    prepares a numpy array to be used as weights for a tf.keras Embed layer downstream

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
        Already fitted on the words in the training corpus
        
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
    
    sample_key=list(embedding_lookup.keys())[0]
    sample_vec=embedding_lookup[sample_key]
    embedding_dim=len(sample_vec)

    
    embeddings_matrix=np.zeros((vocab_size+1,embedding_dim))
    
    for word,encoding in word_index.items():
        embedding_vector=embedding_lookup.get(word)
        if embedding_vector is not None:
            embeddings_matrix[encoding]=embedding_vector
        
    return embeddings_matrix
    

def label_padded_seq_ds(tokenizer,text,label,max_length,padding_type='post',trunc_type='post'):
    """
    uses a tokenizer already fitted on training data to convert list of sentences to sequnces of integer based on the tokenizer's
    word encoding. Then use pad_sequences to either pad or truncate resulting sequences to a uniform max_length. finally combine
    the labels and padded sequences into a tf.Dataset for easy manipulation downstream

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
        Already fitted on the words on the training corpus
    text : str
        text to encode and pad
    label : ndarray/list
        integer class labels of each sentence in the text.
    max_length : int
        length of the resulting sequences from each sentence, wether by padding or truncating.
    padding_type : pad with zeros pre or post
        DESCRIPTION. The default is 'post'.
    trunc_type : drop words wither pre or post
        DESCRIPTION. The default is 'post'.

    Returns
    -------
    ds : tf.Dataset
        tf.Dataset of yielding pairs of (label,seq), whether label is a int64 and seq is a list of length max_length

    """
    seq=tokenizer.texts_to_sequences(text)
    padded_seq=pad_sequences(seq,maxlen=max_length,padding=padding_type,truncating=trunc_type)
    ds=tf.data.Dataset.from_tensor_slices((padded_seq,label))
    return ds

def predicative_sequence_ds(tokenizer,corpus):
    """
    transforms a list of strings (corpus) into a tf.dataset suitable for predicting the next word

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
        a tokenizer already fitted on the corpus
    corpus : list
        list of strings.

    Returns
    -------
    ds : tf.data.Dataset
        a dataset that returns a padded, encoded sequence followed by the encoding of the word that should be predicted.

    """
    inputs=[]
    
    for line in corpus:
        seq=tokenizer.texts_to_sequences([line])[0]
        for i in range(2,len(seq)):
            subseq=seq[:i]
            inputs.append(subseq)
    
    inputs=pad_sequences(inputs)
    ds=tf.data.Dataset.from_tensor_slices(inputs)
    ds=ds.map(lambda x:(x[:-1],x[-1]))
    
    return ds

















