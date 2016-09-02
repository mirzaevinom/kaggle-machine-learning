from __future__ import division
import numpy as np
import pandas as pd
import time, sys
from sklearn.externals import joblib

start = time.time()

train = pd.read_csv( 'train.csv' ) 

test = pd.read_csv( 'test.csv' )


#Parse string image info to 2D array
str_X = train[ train.columns[-1] ].values
X = np.zeros( ( len(str_X) , 96*96 ) )

for nn in xrange( len(str_X) ):
    X[nn] = np.int_( str_X[nn].split(' ') )


str_X = test[ test.columns[-1] ].values

pred_X = np.zeros( ( len(str_X) , 96*96 ) )

for nn in xrange( len(str_X) ):
    pred_X[nn] = np.int_( str_X[nn].split(' ') )


from sklearn.ensemble import RandomForestClassifier

loc_cols = list( train.columns[:-1] )
clf = RandomForestClassifier( n_jobs=-1 , n_estimators=20 )

myarr = np.zeros( (len( pred_X) , len(loc_cols) ) )


filename = 'dataset.joblib'
print "put data in the right layout and map to " + filename
joblib.dump(np.asarray(X, dtype=np.float32, order='F'), filename)
X = joblib.load(filename, mmap_mode='c')
 
for nn in range( len(loc_cols)):
    loc = train[ loc_cols[nn] ].values
    del clf
    clf = RandomForestClassifier( n_jobs=10 , n_estimators=50 )
    clf.fit( X , loc )
    
    myarr[:, nn] = clf.predict( pred_X )
    

out_array = np.zeros( ( len( pred_X)*len(loc_cols)  , 2) )

out_array[:, 0] = np.arange( len( pred_X)*len(loc_cols) ) + 1
out_array[:, 1] = myarr.flatten()

out_csv = pd.DataFrame( out_array , columns=[ 'RowId' , 'Location' ] )
out_csv.to_csv( 'SampleSubmission.csv' , index=False )

end = time.time()
print 'Time elapsed ', round( ( end-start ) / 60 , 2 ) , ' minutes'
    
"""
    
    out_csv = pd.read_csv( 'SampleSubmission.csv' )
    
    
    out_array = myarr.flatten()
    out_csv['Location'][:len(out_array)] = out_array 
    
    out_csv.to_csv( 'SampleSubmission.csv' , index=False )
        
        
    from sklearn import svm    
    x_loc = train[ train.columns[0] ].values[:1000]
    y_loc = train[ train.columns[1] ].values[:1000]
    
    clf_x = RandomForestClassifier( n_jobs=-1 , n_estimators=50 )
    #clf_x = svm.LinearSVC( tol=1e-1 )
    clf_x.fit( X , x_loc )
    
    
    clf_y = RandomForestClassifier( n_jobs=-1 , n_estimators=50 )
    #clf_y = svm.LinearSVC( tol=1e-1 )
    clf_y.fit( X , y_loc )
    
    
    str_X = test[ test.columns[-1] ].values
    
    pred_X = np.zeros( ( len(str_X) , 96*96 ) )
    
    for nn in xrange( len(str_X) ):
        pred_X[nn] = np.int_( str_X[nn].split(' ') )
    
    import matplotlib.pyplot as plt
    
    
    rnum = np.random.randint(0, high=len(test) )
    
    row = np.int_( test.ix[rnum]['Image'].split(' ') )
    
    ex_img = np.reshape( row , (96, 96) )
    
    plt.close('all')
    plt.figure(0)
    
    plt.imshow( ex_img , cmap='Greys' )
    plt.scatter( clf_x.predict(row) , clf_y.predict(row) )"""

