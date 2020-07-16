#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:20:44 2020

helper functions for Sequence modelling in tensorflow

@author: jay
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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


def plot_loss_acc(hist):
  plt.style.use('ggplot')
  
  ax1=plt.subplot(121)
  ax1.plot(hist.history['loss'],label='train',color="red")
  ax1.plot(hist.history['val_loss'],label='val',color="green")
  ax1.set_title("losses")
  ax1.set_xlabel("epochs")
  ax1.set_ylabel("loss")
  ax1.legend()

  ax2=plt.subplot(122)
  ax2.plot(hist.history['mae'],label='train',color="blue")
  ax2.plot(hist.history['val_mae'],label='val',color="orange")
  ax2.set_title("mae")
  ax2.set_xlabel("epochs")
  ax2.set_ylabel("mae")
  ax2.legend()

  plt.tight_layout()
  
def eval(model,x,x_val,window_size,target_size,batch_size,buffer_size,sc):
  num_test=np.ceil(len(x_val)/target_size).astype(int)
  test_start=len(x)-num_test*target_size
  data_start=test_start-window_size

  test_data=x[data_start:]
  test_data=sc.transform(test_data[:,None])[:,0]

  test_ds=data_windowing_3D(test_data,window_size,target_size,batch_size,buffer_size,shuffle=False)
  raw_pred=model.predict(test_ds).flatten()

  scaled_pred=sc.inverse_transform(raw_pred[:,None])[:,0]
  val_pred=scaled_pred[(len(scaled_pred)-len(x_val)):]

  mae=tf.keras.metrics.MAE(val_pred,x_val)
  print("mae: ",mae.numpy().item())

  plt.style.use("seaborn")
  plt.title("val evaluation")
  plt.xlabel("time step")
  plt.ylabel("temperature")
  plt.plot(x_val,color="green",label="true values")
  plt.plot(val_pred,color="orange",label="predicted")
  plt.legend()
  plt.show()
  
  return mae.numpy().item()