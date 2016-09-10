"""

Got some help to generate stats from Thomas Roberts's code

http://www.thomas-robert.fr/en/kaggles-human-or-robot-competition-feedback/

"""


from __future__ import division
import numpy as np
import pandas as pd
import time, sys
from sklearn.externals import joblib
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor

start = time.time()

train = pd.read_csv( 'train.csv' ) 
bids = pd.read_csv( 'bids.csv' ) 

test = pd.read_csv( 'test.csv' )


#==============================================================================
# Data Analysis part
#==============================================================================
def catStats( series ):

    length = len( series )
    counts = series.value_counts()

    nbUnique = counts.count()
    
    hiFreq = 0
    loFreq = 0
    argmax = 0    
    stdFreq = 0

    if len(counts)>0:
        hiFreq = counts[0] / length
        loFreq = counts[-1] / length
        argmax = counts.index[0]    
        stdFreq = np.std( counts / length )
        
    
    return (nbUnique, loFreq, hiFreq, stdFreq, argmax)
    
def timeStats( series ):

     
    series.sort()
    intervals = series[1:].as_matrix() - series[:-1].as_matrix()
    if len(intervals) < 1:
        intervals = np.array([0])
    
    nb = series.shape[0]
    mmin = series.min()
    mmax = series.max()
    mrange = mmax - mmin
    intervalsMin = np.min(intervals)
    intervalsMax = np.max(intervals)
    intervalsMean = np.mean(intervals)
    intervalsStd = np.std(intervals)
    intervals25 = np.percentile(intervals, 25)
    intervals50 = np.percentile(intervals, 50)        
    intervals75 = np.percentile(intervals, 75)

    return (nb, mmin, mmax, mrange,
            intervalsMin, intervalsMax, intervalsMean, intervalsStd,
            intervals25, intervals50, intervals75)        

df = pd.DataFrame()
df['bidder_id'] = bids['bidder_id'].unique()

stat_cols = [ 'auction' , 'device' ,'ip' , 'country' , 'url' ]

for col in stat_cols:

    df = pd.merge( df , 
                  bids.groupby('bidder_id')[ col ].apply(catStats).apply(pd.Series).reset_index() , 
                  on = [ 'bidder_id' ] )
   
    col_list = [ 'nUnq_' + col ,  'loF_'+ col , 'hiF_' +col , 'stdF_'+col, 'argmax_'+col ]

    df = df.rename( columns = dict( zip( range(len(col_list) ) , col_list ) ) )


#==============================================================================
# Time interval computations
#==============================================================================

col_list = [ 'nb', 'mmin', 'mmax', 'mrange',
            'intervalsMin', 'intervalsMax,' 'intervalsMean', 'intervalsStd',
            'intervals25', 'intervals50', 'intervals75' ]

            
df = pd.merge( df , 
              bids.groupby('bidder_id')['time'].apply( timeStats ).apply(pd.Series).reset_index() , 
              on = [ 'bidder_id' ] )            

df = df.rename( columns = dict( zip( range(len(col_list) ) , col_list ) ) )


#Add merchandise type
df = pd.merge( df , 
              bids.groupby('bidder_id')['merchandise'].first().reset_index() ,
              on=['bidder_id'] )

aa = df.dtypes=='object'
strCol_list = aa[aa==True].index.tolist()[1:]

for col in strCol_list:
        
    my_list = df[col].unique()
    
    my_list.sort()
    
    my_dict = dict( zip( my_list, range( len( my_list)  ) ) )
    
    df[ col ] = df[ col ].apply( lambda x : my_dict[x] )


df1 = pd.DataFrame()
df1['avg_freq'] = bids.groupby('bidder_id')['time'].apply( lambda x: ( x.max() - x.min() ) / len(x)  )

df = pd.merge( df, df1.reset_index() , on = ['bidder_id'] )

del df1


df[ df.columns[1:] ] = df[ df.columns[1:] ].apply( lambda x: ( x - x.min() ) / ( x.max() - x.min() ) , axis=0 )


#==============================================================================
# Machine learning part
#==============================================================================
fit_df = pd.merge( df , train,   on=['bidder_id'] ).drop( ['address', 'payment_account' ] , axis=1 )

pred_df = pd.merge( df , test ,   on=['bidder_id'] ).drop( ['address', 'payment_account' ] , axis=1 )


#==============================================================================
# Find the best parameters for RandomForestClassifier using grisearch algorithm
#==============================================================================
from sklearn.grid_search import GridSearchCV

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' , n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [100, 200, 300, 700] ,
    'max_features': ['auto', 'sqrt', 'log2', None] , 
    'max_depth' : [5, 10 , 15 , 20 , None]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit( fit_df[ fit_df.columns[1:-1] ].values , fit_df[ fit_df.columns[-1] ].values )
print CV_rfc.best_params_


#==============================================================================
# Use best params to get the results
#==============================================================================

#clf = DecisionTreeRegressor( )

clf = RandomForestClassifier( n_jobs=-1 , **CV_rfc.best_params_ )

clf.fit( fit_df[ fit_df.columns[1:-1] ].values , fit_df[ fit_df.columns[-1] ].values  )


pred_df['prediction'] = clf.predict( pred_df[ pred_df.columns[1:]].values )

def adjust( row ):
    
    if row>0.1:
        return 1
    else:
        return 0

pred_df['prediction'] = pred_df['prediction'].apply( adjust )

pred_df = pred_df.drop( pred_df.columns[1:-1]  , axis = 1 )

output = pd.DataFrame()

output['bidder_id'] = pd.read_csv( 'sampleSubmission.csv' )['bidder_id']

output = pd.merge( output , pred_df , on = 'bidder_id', how='outer')

output.fillna( 0 , inplace=True)


output.to_csv( 'final_output.csv' , index=False)



    
end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'