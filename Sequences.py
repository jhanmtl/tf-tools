#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:20:44 2020

@author: jay
"""

import tensorflow as tf

def data_windowing_3D(data,window_size,target_size,batch_size,buffer_size,shuffle=True):

  data=tf.expand_dims(data,axis=-1)
  ds=tf.data.Dataset.from_tensor_slices(data)

  ds=ds.window(window_size+target_size,shift=target_size,drop_remainder=True)
  ds=ds.flat_map(lambda w: w.batch(window_size+target_size))
  if shuffle:
    ds=ds.shuffle(buffer_size)
  ds=ds.map(lambda w: (w[:window_size],tf.squeeze(w[window_size:],axis=-1)))
  ds=ds.batch(batch_size).prefetch(1)

  return ds
