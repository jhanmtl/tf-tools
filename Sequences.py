#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:20:44 2020

@author: jay
"""

import tensorflow as tf
import matplotlib.pyplot as plt

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
  
def pred_and_eval(model,scaler,x_val,x_all_normalized,window_size,target_size,batch_size,buffer_size):
  total_ds=data_windowing_3D(x_all_normalized,window_size,target_size,batch_size,buffer_size,shuffle=False)
  pred_all=model.predict(total_ds)                                                             
  pred_all=scaler.inverse_transform(pred_all)[:,0]                                                  
  val_start=len(pred_all)-len(x_val)                                                            
  pred_val=pred_all[val_start:].flatten()                                                       

  mae=tf.keras.metrics.MAE(pred_val,x_val)

  

  plt.title("val prediction")
  plt.style.use("seaborn")
  plt.plot(x_val,color="green",label="val")
  plt.plot(pred_val,color="orange",label="predicted")
  plt.xlabel("val time step")
  plt.ylabel("temperature")
  plt.legend()
  plt.show()

  return mae.numpy().item()