# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 19:17:12 2016

@author: Inom Mirzaev

"""

from __future__ import division

import numpy as np
import os, cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

clahe = cv2.createCLAHE()




ROWS = 90
COLS = 160
CHANNELS = 3


def image_process(fname, COLS=COLS, ROWS=ROWS):
    
    img  = cv2.imread( fname ,  cv2.IMREAD_COLOR )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img[:,:,0] = cv2.GaussianBlur(img[:,:,0], (5, 5) , 0 )
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
 
    img = cv2.resize( img , ( COLS , ROWS )  , interpolation=cv2.INTER_CUBIC)

    return img       
    


if __name__=='__main__':
    
    train = []
    
    
    for folder in os.listdir( 'train' ):
        for fname in os.listdir( 'train/'+folder ):
            train.append( [folder, 'train/'+folder+'/'+fname] )
    
    
    train = pd.DataFrame( train, columns = ['type' , 'path'] )
    
        
    X_all = np.zeros( [ len(train) ,  ROWS, COLS, CHANNELS ] )
        
    for nn in xrange( len(train) ):
        
        fname = train['path'][nn]
        
        X_all[nn] = image_process( fname ) 
     
    y_all = LabelEncoder().fit_transform( train['type'].values )    
    np.save('input/X_all' , X_all)
    np.save('input/y_all' , y_all)
    del X_all, y_all, train    
 
    
    #==============================================================================
    # Do the same thing for the test data
    #==============================================================================
        
    
       
    test_data = os.listdir( 'test_stg1' )
 
    
    X_test = np.zeros( [ len(test_data) ,   ROWS, COLS, CHANNELS ] )
    
    for nn in xrange( len(test_data) ):
        
        fname = 'test_stg1/'+test_data[nn]
        X_test[nn] = image_process( fname )     
    
    np.save('input/X_test' , X_test)
    
    del test_data, X_test
    

