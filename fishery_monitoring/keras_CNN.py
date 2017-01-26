# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:44:26 2016

@author: Inom Mirzaev

"""
from __future__ import division

import numpy as np
import time , os
import pandas as pd

from keras.models import Sequential

from keras.layers import Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D , ZeroPadding2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping

from rgb_process import ROWS, COLS , CHANNELS

from keras import backend as K



def simple_CNN( n_classes , ROWS=ROWS, COLS=COLS, CHANNELS=CHANNELS):
    
    """
    This is a CNN from 
    https://www.kaggle.com/jeffd23/the-nature-conservancy-fisheries-monitoring/
    deep-learning-in-the-deep-blue-lb-1-279/notebook
    """
       
    def center_normalize(x):
        return (x - K.mean(x)) / K.std(x)
    
    model = Sequential()
    
    model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))
    
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
    
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    
    model.add( Dense( n_classes ) )
    model.add( Activation('sigmoid') )
    
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4))
      
    return model

def deep_CNN(n_classes , ROWS=ROWS, COLS=COLS, CHANNELS=CHANNELS):
    
    """
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    """
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(ROWS, COLS, CHANNELS)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_classes, activation='sigmoid'))
    
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')


    return model

    
if __name__=='__main__':
    
    start = time.time()

    print time.strftime( '%H:%M', time.localtime() )

    #train = pd.read_csv( 'train.csv' )
    #test = pd.read_csv( 'test.csv' )
    #X_all = train[train.columns[2:-1]].values    
    #y_all = train[train.columns[-1]].values
    #X_test = test[test.columns[1:]].values
    #X_all = X_all.reshape( len(X_all),  ROWS, COLS, CHANNELS).astype('float32')
    #X_test = X_test.reshape( len(X_test),  ROWS, COLS, CHANNELS).astype('float32')
    
    X_all = np.load('input/X_all.npy')
    X_test = np.load('input/X_test.npy')
    y_all = np.load('input/y_all.npy')
    
    FISH_CLASSES = os.listdir( 'train' )
    
    from keras.utils.np_utils import to_categorical
    y_all = to_categorical(y_all , len(FISH_CLASSES) )  
    
    # normalize from [0, 255] to [0, 1]
    #X_all /= 255.0
    #X_test /= 255.0
    
    #model = deep_CNN( len(FISH_CLASSES) )
    model = simple_CNN( len(FISH_CLASSES) )
    
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
        
    model.fit(X_all, y_all, batch_size=64, nb_epoch=1,
              validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])

    
    test_preds = model.predict( X_test , verbose=1 )
    
    submission = pd.DataFrame(test_preds, columns=FISH_CLASSES )
    
    submission.insert(0, 'image', os.listdir( 'test_stg1' ) )
    
    submission.to_csv('output/submission_CNN.csv' , index=False)
    
    
    end = time.time()
    
    print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'





