# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 2016

@author: Inom Mirzaev
"""

import numpy as np
import pandas as pd
import time

start = time.time()

train = pd.read_csv( 'train.csv' ) 

test = pd.read_csv( 'test.csv' )

X = train[ train.columns[1:] ].values

y = train[ train.columns[0] ].values


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#clf = svm.LinearSVC( tol=1e-3, C = 100) 
clf = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
clf.fit( X , y)

p_labels = clf.predict( test.values )

out_array = np.zeros( ( len(p_labels) , 2 ) )

out_array[:, 0] = np.arange( 1 , len(p_labels) + 1 )
out_array[:, 1] = p_labels

out_csv = pd.DataFrame( out_array , columns=['ImageId' , 'Label'] , dtype='int32')

out_csv.to_csv( 'sample_submission.csv' , index=False)

end = time.time()


print 'Time elapsed ', round( end-start, 0 ) , ' second'
"""
import matplotlib.pyplot as plt
rnum = np.random.randint(0, high=len(test) )
row = test.ix[rnum].values
guess = clf.predict( row )

ex_img = np.reshape( row , (28, 28) )

plt.close('all')
plt.figure(0)

plt.imshow( ex_img , cmap='Greys' )

plt.title( str( guess[0] ) )"""

