#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:39:18 2020

@author: jay
"""

import matplotlib.pyplot as plt

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