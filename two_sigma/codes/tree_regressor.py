from __future__ import division

import kagglegym
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import  LinearSVR

from sklearn.ensemble import GradientBoostingRegressor , ExtraTreesRegressor
import xgboost as xgb

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Get the train dataframe
train = observation.train.copy()

#==============================================================================
# Replace NaN values with average of the mean values grouped by 'id'
#==============================================================================
#cols_used = [ 'technical_20'   , 'technical_30'  , 'fundamental_11' , 'fundamental_18'  ] 

mean = train.groupby('timestamp').transform('mean')
train.fillna(mean, inplace=True)

mean = train.groupby('id').transform('mean')
train.fillna(mean, inplace=True)

poly_dim = 2
model_select = 4
#==============================================================================
# Choose one the most correlated two columns
#==============================================================================


Y = train['y'].values

corr_dict = {}

feature_cols = list( set(train.columns) - {'id' ,'y' , 'timestamp'} )

for col in feature_cols:
    X = np.nan_to_num( train[col].values )
    X = np.diff(X)
    X = np.diff(X)
    corr_dict[col] = np.abs( np.corrcoef(X , Y[2:] )[0, 1] )

sorted_corr = sorted(corr_dict, key=corr_dict.get)  

cols_used  = sorted_corr[-2:]
#cols_used = ['technical_20']
# Observed with histograms:
sigma = np.std( Y - np.mean(Y) )    
low_y_cut = -0.075#-3*sigma
high_y_cut = 0.075#3*sigma


y_values_within = (~(train.y > high_y_cut) & ~(train.y < low_y_cut) )

ymean_dict = dict(train.groupby(["id"])["y"].mean())
ymedian_dict = dict(train.groupby(["id"])["y"].median())


def get_weighted_y(series):
    col, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymean_dict[col] if col in ymean_dict else y
    #return 0.95 * y + 0.05 * ymedian_dict[col] if col in ymedian_dict else y
    
    
X_train = train.loc[ y_values_within , cols_used ].values

#Scale the Train set, use same values to scale test set
mean_values = np.nanmean( X_train , axis=0 )
mean_values[ np.isnan( mean_values ) ] = 0
            
std_values  = np.nanstd( X_train , axis=0 )
std_values[ np.isnan( std_values ) ] = 1
 
X_train  = (X_train - mean_values) / std_values
X_train = np.nan_to_num( X_train )

poly = PolynomialFeatures( degree=poly_dim ,  interaction_only=False )

X_train = poly.fit_transform( X_train )

y_train =  train.loc[ y_values_within , 'y'].values



model1 = lm.LinearRegression( n_jobs=4  , fit_intercept=False )
model2 = lm.Ridge( alpha=1e5, tol=1e-4 , fit_intercept=False )

model3 = LinearSVR(C=1e-5, loss='squared_epsilon_insensitive', 
                   tol=1e-4, fit_intercept=False, dual=False )

#model4 = xgb.XGBRegressor(reg_alpha = 0.001 , reg_lambda=1e4,
#                          learning_rate=0.1 , n_estimators=500 , nthread=5)

print('Training linear regressors...')
model1.fit( X_train , y_train )
model2.fit( X_train , y_train )
model3.fit( X_train , y_train )





if model_select==4:
    print('Training Decision Tree regressor...')
   
    tree_cols = feature_cols
    
    train = observation.train.copy()
     
    train = train[ tree_cols ]
    
    #d_mean = train.mean(axis=0)
    d_mean= train.median(axis=0)
    
    n = train.isnull().sum(axis=1)
    
    for c in train.columns:
        train.loc[:, c + '_nan_'] = pd.Series( train[c].isnull().values.astype(int) , index = train.index )
        d_mean[c + '_nan_'] = 0
        
    train = train.fillna(d_mean)
    train['znull'] = n
    n = []
    
    model4= ExtraTreesRegressor(n_estimators=20, max_depth=4, n_jobs=-1, random_state=308537, verbose=0)
    model4.fit( train.values , Y )

del train

print('Predicting...')
while True:
    
    test = observation.features.copy()
    for col in cols_used:
        mean = test.groupby('timestamp')[col].transform('mean')
        test[col].fillna(mean, inplace=True)
        
        mean = test.groupby('id')[col].transform('mean')
        test[col].fillna(mean, inplace=True)
    
    X_test = test[cols_used].values
    X_test  = (X_test - mean_values) / std_values
    X_test = np.nan_to_num( X_test )
    
    X_test = poly.fit_transform( X_test )
    
    
    if model_select==0:
        y_pred = ( model1.predict( X_test ).clip(low_y_cut, high_y_cut ) +\
                 model2.predict( X_test ).clip(low_y_cut, high_y_cut ) + \
                 model3.predict( X_test ).clip(low_y_cut, high_y_cut ) ) / 3.0# + \
                 #model4.predict( X_test ).clip(low_y_cut, high_y_cut) ) / 4.0
                 
                 
    elif model_select==1:
        y_pred = model1.predict( X_test )
        
    elif model_select==2:
        y_pred = model2.predict( X_test )
        
    elif model_select==3:
        y_pred = model3.predict( X_test )
        
    elif model_select==4:
        
        test = observation.features[ tree_cols ].copy()
        n = test.isnull().sum(axis=1)
        
        for c in tree_cols:
            test.loc[:, c + '_nan_'] = pd.Series( test[c].isnull().values.astype(int) , index = test.index )
            
        test = test.fillna(d_mean)
        test['znull'] = n
        n = []
        
        y_pred = ( model3.predict( X_test ) + model4.predict( test.values ) ) / 2
    
#        y_pred = ( model1.predict( X_test ) +\
#                   model2.predict( X_test )+ \
#                   model3.predict( X_test )  + \
#                   model4.predict( test.values ) ) / 4.0
#        
        
    observation.target.y = y_pred.clip(low_y_cut, high_y_cut)
    
    ## weighted y using average value
    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)
    
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp) )

    observation, reward, done, info = env.step(target)
    if done:
        print("Public score: {}".format( info["public_score"] ) )
        break
