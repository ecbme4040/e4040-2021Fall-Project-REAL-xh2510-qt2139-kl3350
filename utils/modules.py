#Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import random

#Attention module with encoder depth 
# r: num of residual units in Down/Up sampling
# p: num of residual units before and after Soft Mask Branch and Trunk Branch
# t: num of residual units in the Trunk Branch

#Define Residual Unit
def residual_unit(X, filter_num_first, filter_num_final, stride = 1, short_cut=True):

    ShortCut = layers.Conv2D(filter_num_final, (1, 1), padding='same', strides=stride)(X)

    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_first, (1,1), padding='same')(X)

    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_first, (3,3), padding='same', strides = stride)(X)
    
    X = layers.BatchNormalization()(X)
    X = layers.Activation('relu')(X)
    X = layers.Conv2D(filter_num_final, (1,1), padding='same')(X)
    
    if short_cut == True:
        X = layers.Add()([X, ShortCut])
    
    return X


def attention_module(X, filter_num_first, filter_num_final, depth=1):
    
    #Input
    X = residual_unit(X, filter_num_first, filter_num_final, short_cut=True) # p = 1
    
    #Trunk Branch: t = 2
    Trunk = residual_unit(X, filter_num_first, filter_num_final, short_cut=True)
    Trunk = residual_unit(Trunk, filter_num_first, filter_num_final, short_cut=True)

    #Soft Mask Branch
    ##Down Sampling: r = 1
    Mask = layers.MaxPool2D(pool_size=(2, 2), padding='same')(X)
    Mask = residual_unit(Mask, filter_num_first, filter_num_final, short_cut=True)
    
    skip_connection = []
    for i in range(depth - 1):
      
      skip_connection_list = residual_unit(Mask, filter_num_first, filter_num_final, short_cut=True)
      skip_connection.append(skip_connection_list)
      
      Mask = layers.MaxPool2D(pool_size=(2, 2), padding='same')(Mask)
      Mask = residual_unit(Mask, filter_num_first, filter_num_final, short_cut=True) # r = 1
    
    skip_connection_reverse = list(reversed(skip_connection))

    ##Up Sampling: r = 1
    for i in range(depth - 1): 
      
      Mask = residual_unit(Mask, filter_num_first, filter_num_final, short_cut=True) # r = 1
      Mask = layers.UpSampling2D()(Mask) 

      Mask = layers.Add()([Mask, skip_connection_reverse[i]])

    Mask = residual_unit(Mask, filter_num_first, filter_num_final, short_cut=True)
    Mask = layers.UpSampling2D()(Mask)   
    
    Mask = layers.BatchNormalization()(Mask)
    Mask = layers.Activation('relu')(Mask)
    Mask = layers.Conv2D(filter_num_final, (1,1), padding='same')(Mask)
    Mask = layers.BatchNormalization()(Mask)
    Mask = layers.Activation('relu')(Mask)
    Mask = layers.Conv2D(filter_num_final, (1,1), padding='same')(Mask)
    Mask = layers.Activation('sigmoid')(Mask)
    
    #Implement: H = (1 + M) * F
    Mask = layers.Lambda(lambda x: x + 1)(Mask)
    Out = layers.Multiply()([Mask, Trunk]) 
     
    Out = residual_unit(Out, filter_num_first, filter_num_final, short_cut=True) # p = 1

    return Out

