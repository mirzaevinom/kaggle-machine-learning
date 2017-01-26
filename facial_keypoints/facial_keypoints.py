from __future__ import division
import numpy as np
import pandas as pd
import time, os
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

start = time.time()

print time.strftime( '%H:%M', time.localtime() )

def histogram_equalization( image , number_bins=256):
    """This transform flattens the graylevel histogram of an image so that 
    all intensities are as equally common as possible. For more information see,
    
    http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html"""

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


#============================================================================== 
# Get images in string format and write them in to an array
#==============================================================================

# Load train.csv
train = pd.read_csv( 'train.csv' ) 
train['Image'] = train['Image'].apply( lambda im: np.fromstring( im , sep=' ') )
#Apply histogram equalization
train['Image'] = train['Image'].apply( histogram_equalization )

print len(train['Image'][0])

# Store in 2D array
X = np.vstack( train['Image'].values )
X = X.astype( np.float32 )

# Load test.csv
test = pd.read_csv( 'test.csv' )
test['Image'] = test['Image'].apply( lambda im: np.fromstring( im , sep=' ') )

#Apply histogram equalization
test['Image'] = test['Image'].apply( histogram_equalization )

#Store in a 2D array
pred_X = np.vstack( test['Image'].values )
pred_X = pred_X.astype( np.float32 )


#==============================================================================
# Fill the NaNs in location columns with the mean of that column
#==============================================================================

loc_cols = list( train.columns[:-1] )

for col in loc_cols:
    
    train[col].fillna( train[col].mean() , inplace=True )


myarr = np.zeros( (len( pred_X) , len(loc_cols) ) )


#Creates memory mapped error. This is to prevent memory leak for multiple jobs
filename = 'dataset.joblib'
print "put data in the right layout and map to " + filename
joblib.dump(np.asarray(X, dtype=np.float32, order='F'), filename)
X = joblib.load(filename, mmap_mode='c')
 
#For each column of train.csv train and predict the values for test.csv
for nn in range( len(loc_cols)):
    loc = train[ loc_cols[nn] ].values.astype( np.float32 ).flatten()
    
    """clf = RandomForestClassifier( n_jobs=10 , n_estimators=10 , 
                                  max_depth=8 , max_features=None )"""
    clf = DecisionTreeRegressor( max_depth = 8 )                              
    clf.fit( X , loc )
    
    myarr[:, nn] = clf.predict( pred_X ).copy()
    del clf

out_array = np.zeros( ( len( pred_X)*len(loc_cols)  , 2) )

out_array[:, 0] = np.arange( len( pred_X)*len(loc_cols) ) + 1
out_array[:, 1] = myarr.flatten()

out_csv = pd.DataFrame( out_array , columns=[ 'RowId' , 'Location' ] )
out_csv.to_csv( 'SampleSubmission.csv' , index=False )

for file in os.listdir("./"):
    if ( file.find('dataset') >= 0 ):
        try:        
            os.remove( file )
        except OSError, e:
            print ("Error: %s - %s." % (e.filename,e.strerror))
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

