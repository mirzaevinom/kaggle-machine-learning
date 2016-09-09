from __future__ import division
import numpy as np
import pandas as pd
import time, sys
from sklearn.externals import joblib

start = time.time()

train = pd.read_csv( 'train.csv' ) 
bids = pd.read_csv( 'bids.csv' ) 

test = pd.read_csv( 'test.csv' )


df = pd.DataFrame()
df['num_auc'] = bids.groupby('bidder_id')['auction'].nunique()
df['num_device'] = bids.groupby('bidder_id')['device'].nunique()
df['num_ip'] = bids.groupby('bidder_id')['ip'].nunique()
df['num_time'] = bids.groupby('bidder_id')['time'].nunique()
df['num_country'] = bids.groupby('bidder_id')['country'].nunique()
df['num_merch'] = bids.groupby('bidder_id')['merchandise'].nunique()
df['num_url'] = bids.groupby('bidder_id')['url'].nunique()


def time_freq(df):
    
    arr = df['time'].values
    
    return ( np.max(arr) - np.min(arr) ) / len(arr) / 1e10
    

df['avg_freq'] = bids.groupby('bidder_id').apply( time_freq )


df = df.reset_index()

#common = pd.merge( df ,  train, on='')
    
end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'