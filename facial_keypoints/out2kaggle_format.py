from __future__ import division

import pandas as pd
import numpy as np

sample = pd.read_csv('SampleSubmission.csv')

train = pd.read_csv( 'train.csv' ) 

cols = train.columns[:-1]

data = np.reshape( sample['Location'].values , (-1, 30) )
imageid = 30 * range(1, len(data)+1 )

imageid.sort()

"""
out_df = pd.DataFrame( data, columns = cols )


out_df = out_df.interpolate( method ='nearest' , axis=0 )

out_df.fillna( method='bfill', inplace=True)
out_df.fillna( method='ffill', inplace=True)


sample['Location'] = out_df.values.flatten()"""
sample['RowId'] = sample['RowId'].astype('int')
sample['ImageId'] = imageid
sample['FeatureName'] = len(data) * list(cols)

cols = list( sample.columns )
sample = sample[ [cols[0]] + cols[2:] + [cols[1]] ]

idlook = pd.read_csv('IdLookupTable.csv')

key_col = ['ImageId', 'FeatureName']

df_with_idx = sample.reset_index()
common = pd.merge(df_with_idx, idlook , on=key_col)['index']
mask = df_with_idx['index'].isin( common )

result =  df_with_idx[mask].drop('index',axis=1)
result['RowId'] = np.arange(len(result)) +1
result = result.drop( key_col , axis=1)
result = result.reset_index( drop=True)
result.to_csv( 'final_output.csv', index=False)

