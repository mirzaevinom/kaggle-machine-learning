from __future__ import division

import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor , AdaBoostRegressor , RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.preprocessing import PolynomialFeatures

import time


start = time.time()

print time.strftime( '%H:%M', time.localtime() )


# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train

#cols_used = []
#for c in train.columns:
#    if train[c].isnull().sum()<len(train)/200: cols_used.append(c)
#
#cols_used= list( set(cols_used) - {'y', 'id' , 'timestamp'} )
                


# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

y_values_within = ((train['y'] > low_y_cut) & (train['y'] <high_y_cut))


cols_used = ['technical_20']
#cols_used = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']
#cols_used = []
#for c in train.columns:
#    if 'technical' in c: cols_used.append( c )

    
print("Training...")
model = lm.LinearRegression( n_jobs=2 , normalize=False )
#model = lm.Ridge(alpha=1e5)
#model = LinearSVR(C=1e-3, loss='squared_epsilon_insensitive', 
#                   tol=1e-4, fit_intercept=False)
#model = SVR(C=1e-4, shrinking=False , tol=1e-4)

#model = lm.BayesianRidge( n_iter=1000 , tol=1e-3)
#model =  GradientBoostingRegressor( loss='huber', max_depth=3, learning_rate=0.1,
#                                   max_features=None , n_estimators=10 )

#model =  RandomForestRegressor( criterion ='mse'  , max_depth=3 , 
#                               n_jobs=2 , max_features=None, n_estimators=10 )

X_train = train.loc[y_values_within, cols_used].values

#Scale the Train set, use same values to scale test set
mean_values = np.nanmean( X_train , axis=0 )
mean_values[ np.isnan(mean_values) ] = 0
            
std_values  = np.nanstd( X_train , axis=0)
std_values[np.isnan(std_values)] = 1
 
X_train  = (X_train - mean_values) / std_values
X_train = np.nan_to_num( X_train )

poly = PolynomialFeatures( degree=1, interaction_only=True )

X_train = poly.fit_transform( X_train )


y_train =  train.loc[y_values_within, 'y'].values

model.fit(X_train , y_train)

print("Predicting...")
while True:
    
    test = observation.features
    
    X_test = test[cols_used].values
    X_test  = (X_test - mean_values) / std_values
    X_test = np.nan_to_num( X_test )
    X_test = poly.fit_transform( X_test )
    
    observation.target.y = model.predict(X_test).clip(low_y_cut, high_y_cut)
       
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp) )

    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format( info["public_score"] ) )
        break
    
end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'
