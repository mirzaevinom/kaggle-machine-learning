# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:17:12 2016

@author: Inom Mirzaev

"""

from __future__ import division

import numpy as np
import os, cv2
import pandas as pd

clahe = cv2.createCLAHE()




ROWS = 135
COLS = 240


def image_process(fname, COLS=COLS, ROWS=ROWS):
    
    img  = cv2.imread( fname , 0)
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5, 5) , 0 )
    
    #img = cv2.adaptiveThreshold( img , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 11 , 15 )
    ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    #img[:,:] = 255
    #cv2.drawContours(img, contours ,  -1, (0,255,0), 3) 

    img = cv2.resize( img , ( COLS , ROWS )  , interpolation=cv2.INTER_NEAREST)

    return img       
    


if __name__=='__main__':
    
    train_data = []
    
    
    for folder in os.listdir( 'train' ):
        for fname in os.listdir( 'train/'+folder ):
            train_data.append( [folder, 'train/'+folder+'/'+fname] )
    
    
    train = pd.DataFrame( train_data, columns = ['type' , 'path'] )
    
    del train_data 
    
    
    train_data = np.zeros( [ len(train) ,  COLS*ROWS +1 ] )
    
    
    type_list  = os.listdir( 'train' )
    type2num = dict( zip( type_list , range(len(type_list) )  ) )
    
    for nn in xrange( len(train) ):
        
        fname = train['path'][nn]
        
        img = image_process( fname )
       
        train_data[nn, :-1] = np.ravel( img )
        train_data[nn, -1] = type2num[ train['type'][nn] ]
    
    df = pd.DataFrame( train_data )
    
    train = train.join(df)
    
    del train_data
    
    train.to_csv('input/train.csv' , index=False)
    
    del train
    
    
    
    #==============================================================================
    # Do the same thing for the test data
    #==============================================================================
        
    
       
    test_data = os.listdir( 'test_stg1' )
    test= pd.DataFrame( test_data, columns = ['path'] )
    
    del test_data 
    
    test_data = np.zeros( [ len(test) ,  COLS*ROWS ] )
    
    for nn in xrange( len(test) ):
        
        fname = 'test_stg1/'+test['path'][nn]
        img = image_process( fname )     
        test_data[nn] = np.ravel( img )
        
    
    df = pd.DataFrame( test_data )
    
    test = test.join(df)
    
    del test_data
    
    test.to_csv('input/test.csv' , index=False)

