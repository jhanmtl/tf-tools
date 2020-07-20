#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:36:40 2020
helper functions for NLP modelling in tensorflow

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

def predicative_sequence_ds(tokenizer,corpus,batch_size,buffer_size):
    """
    transforms a list of strings (corpus) into a tf.dataset for predicting next word based on previous words.
    performs integer encoding with supplied tokenizer, and then padding to max sentence length. for each encoded
    and padded sentence sequence, then start out with a subsequence of the first two word encoding of the sequence, and then
    iteratively grow this subsequence until encompasses original sequence. each subsequence padded to the original maxlength.
    

    Parameters
    ----------
    tokenizer : tf.keras.preprocessing.text.Tokenizer
        already fitted on corpus
    corpus : list
        list of strings, NOT separated into individual words per list entry
    batch_size : integer
        batch size of resulting tf.dataset.
    buffer_size : integer
        buffer size of resulting tf.dataset.

    Returns
    -------
    ds : tf.Dataset
        tf.Dataset of yielding pairs of (seq,label), whether seq is a sequence of integers based on tokenizer encoding and
        padded to a uniform length determined internally, and label is an integer representing the encoding of the next
        word in the sequence

    """

    seq=tokenizer.texts_to_sequences(corpus)
    padded_seq=pad_sequences(seq,maxlen=None,padding='pre')
    max_length=padded_seq.shape[1]
    
    child_seq=[]
    
    for subseq in padded_seq:
      boundary=np.count_nonzero(np.logical_not(subseq>0))+2
      while boundary<len(subseq):
        extract=subseq[:boundary]
        padded_extract=pad_sequences([extract],maxlen=max_length)[0].tolist()
        child_seq.append(padded_extract)
        boundary+=1
    
    child_seq=np.array(child_seq)
    
    total_seq=np.concatenate((padded_seq,child_seq),axis=0)
    ds=tf.data.Dataset.from_tensor_slices(total_seq)
    ds=ds.batch(batch_size)
    ds=ds.shuffle(buffer_size)
    ds=ds.map(lambda x: (x[:,:-1],x[:,-1]))
    
    return ds

















