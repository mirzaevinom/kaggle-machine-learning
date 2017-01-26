from __future__ import division

import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


env = kagglegym.make()
observation = env.reset()


train = observation.train
train = train[cols_used]

#d_mean = train.mean(axis=0)
d_mean= train.median(axis=0)


train = observation.train[cols_used]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
    
train = train.fillna(d_mean)
train['znull'] = n
n = []

rfr = ExtraTreesRegressor(n_estimators=4, max_depth=4, n_jobs=4, random_state=308537, verbose=0)
model1 = rfr.fit(train, observation.train['y'])


train = observation.train
mean = train.groupby('timestamp').transform('mean')
train.fillna(mean, inplace=True)

mean = train.groupby('id').transform('mean')
train.fillna(mean, inplace=True)

poly_dim = 2
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



model2 = LinearRegression( n_jobs=4  , fit_intercept=False )

model2.fit( X_train , y_train )
train = []

#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(observation.train.groupby(["id"])["y"].median())
def get_weighted_y(series):
    id, y = series["id"], series["y"]
    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y

i = 0; reward_=[]
while True:
    
    test = observation.features[cols_used]
    n = test.isnull().sum(axis=1)
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = n
    
    
    pred = observation.target

    test1 = observation.features
    for col in cols_used:
        mean = test.groupby('timestamp')[col].transform('mean')
        test1[col].fillna(mean, inplace=True)
        
        mean = test.groupby('id')[col].transform('mean')
        test1[col].fillna(mean, inplace=True)
    
    X_test = test1[cols_used].values
    X_test  = (X_test - mean_values) / std_values
    X_test = np.nan_to_num( X_test )
    
    X_test = poly.fit_transform( X_test )
    
    pred['y'] = ( model1.predict(test).clip(low_y_cut, high_y_cut) +\
                  model2.predict(X_test).clip(low_y_cut, high_y_cut) ) / 2
    
    pred['y'] = pred.apply(get_weighted_y, axis = 1)
    o, reward, done, info = env.step(pred[['id','y']])
    reward_.append(reward)
    
    if i % 100 == 0:
        print("Timestamp #{}".format(i) )

    i += 1
    if done:
        print("Public score: {}".format( info["public_score"] ) )
        break