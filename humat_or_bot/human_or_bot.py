from __future__ import division
import numpy as np
import pandas as pd
import time, sys
from sklearn.externals import joblib

start = time.time()

train = pd.read_csv( 'train.csv' ) 
bids = pd.read_csv( 'bids.csv' ) 

test = pd.read_csv( 'test.csv' )


