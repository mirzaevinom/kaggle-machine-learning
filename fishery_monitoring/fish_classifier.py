# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:44:26 2016

@author: Inom Mirzaev

"""
from __future__ import division

import numpy as np
import time
import pandas as pd


from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

start = time.time()

print time.strftime( '%H:%M', time.localtime() )

# Load train and test set
train = pd.read_csv( 'input/train.csv' )
test = pd.read_csv( 'input/test.csv' )


FISH_CLASSES = train['type'].unique().tolist()


X_all = train[train.columns[2:-1]].values
y_all = train[train.columns[-1]].values

#from keras.utils.np_utils import to_categorical
#y_all = to_categorical(y_all , len(FISH_CLASSES) )  
    


"""
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=0.25, 
                                                    stratify=y_all)"""


X_test = test[test.columns[1:]].values

#model = RandomForestClassifier( n_estimators=1000, n_jobs=4 , max_depth=100 , max_features=10000 )

print 'Start parameter searching'

"""param_grid = { 'n_estimators':[10],
              "max_depth": [ 40, 60],
              'max_features': [ 1000 , 2000, 3000] }

clf = GradientBoostingClassifier()

grid = GridSearchCV( clf , param_grid,
                    cv=ShuffleSplit(n=len(X_all), n_iter=3, test_size=0.2),
                    scoring='log_loss',
                    n_jobs=6 ).fit( X_all , y_all  )

param_dict = grid.best_params_
print 'Best params: ', param_dict
#print type(param_dict)
param_dict['n_estimators']=300

print 'Score for the best classifier: ' , grid.best_score_ 

#model = grid.best_estimator_ 
 

model = GradientBoostingClassifier(**param_grid)"""

model = GradientBoostingClassifier( n_estimators = 300 , max_depth=40 , max_features = 4000)
                    
model.fit( X_all , y_all )   


test_preds = model.predict_proba( X_test )

submission = pd.DataFrame(test_preds, columns=FISH_CLASSES )

submission.insert(0, 'image', test['path'].values )
submission.to_csv('output/submission_otsu.csv' , index=False)

end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'





