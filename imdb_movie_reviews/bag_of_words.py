from __future__ import division
import numpy as np
import pandas as pd
import time, os
from sklearn.ensemble import RandomForestClassifier ,  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import ShuffleSplit

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals import joblib

from bs4 import BeautifulSoup 

import re


#import nltk
#nltk.download()  # Download text data sets, including stop words

from nltk.corpus import stopwords

start = time.time()


train = pd.read_csv('labeledTrainData.tsv', sep='\t')

test = pd.read_csv('testData.tsv', sep='\t')


stops = set( stopwords.words( "english" ) )


def html_parser(row , stops = stops ):
    mystr = BeautifulSoup(row).get_text()
    
    words = set ( re.sub("[^a-zA-Z]",          
                      " ",                      
                      mystr ).lower().split() )
                      
                  
    meaningful_words = list( words - stops ) 
    
    return  " ".join( meaningful_words )

#Remove html tags, remove stop words like a, the, ...    
train['review'] =train['review'].apply( html_parser )
test['review'] = test['review'].apply( html_parser )



vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

#Vectorize each review
train_data_features = vectorizer.fit_transform( train['review'].values ).toarray()
#Creates memory mapped train data
filename = 'dataset.joblib'
print "put data in the right layout and map to " + filename
joblib.dump(np.asarray( train_data_features , dtype=np.float32, order='F'), filename)
train_data_features = joblib.load(filename, mmap_mode='c')

test_data_features = vectorizer.fit_transform( test['review'].values ).toarray()

#==============================================================================
# Search for optimized paramaters using GridSearchCV
#==============================================================================
param_grid = { 'n_estimators':[100],
"max_depth": [3, 5, 10],
'max_features': [  'log2', 'sqrt' ,  None] }

gbr = GradientBoostingRegressor()

grid = GridSearchCV( gbr , param_grid,
cv=ShuffleSplit(n=len(train_data_features), n_iter=10, test_size=0.25),
scoring="mean_squared_error",
n_jobs=10 ).fit( train_data_features , train['sentiment'].values  )


clf = grid.best_estimator_ 

#==============================================================================
# Use best params to get the results
#==============================================================================


#clf = RandomForestClassifier( n_estimators=100 , n_jobs=10 , max_features=None )

#clf = GradientBoostingRegressor( n_estimators=100 )

clf.fit( train_data_features , train['sentiment'].values  )


pred_df = pd.read_csv('sampleSubmission.csv')

pred_df['sentiment'] = clf.predict(  test_data_features )

def adjust( row ):
    
    if row>0.1:
        return 1
    else:
        return 0  

pred_df['sentiment'] = pred_df['sentiment'].apply( adjust )

pred_df.to_csv( 'final_output.csv' , index=False)


#==============================================================================
# Delete memory mapped files
#==============================================================================
for file in os.listdir("./"):
    if ( file.find('dataset') >= 0 ):
        try:        
            os.remove( file )
        except OSError, e:
            print ("Error: %s - %s." % (e.filename,e.strerror))
            
        
end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'