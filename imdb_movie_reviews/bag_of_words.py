from __future__ import division
import numpy as np
import pandas as pd
import time, os
from sklearn.ensemble import RandomForestClassifier ,  GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import ShuffleSplit

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.externals import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from bs4 import BeautifulSoup 

import re


#import nltk
#nltk.download()  # Download text data sets, including stop words


start = time.time()


train = pd.read_csv('labeledTrainData.tsv', sep='\t' , quoting = 3)

unlab_train = pd.read_csv('unlabeledTrainData.tsv', sep='\t', quoting = 3)


test = pd.read_csv('testData.tsv', sep='\t' , quoting = 3)



from nltk.corpus import stopwords

stops = set( stopwords.words( "english" ) )


def html_parser(row , stops = stops ):
    mystr = BeautifulSoup(row).get_text()
    
    words = set ( re.sub("[^a-zA-Z]",          
                      " ",                      
                      mystr ).lower().split() )
                      
                  
    meaningful_words = list( words - stops ) 
    
    return  " ".join( meaningful_words )

#Remove html tags, remove stop words like a, the, ...    
train['review']     = train['review'].apply( html_parser )
unlab_train['review']     = unlab_train['review'].apply( html_parser )

test['review']      = test['review'].apply( html_parser )

"""
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)""" 


vectorizer = TfidfVectorizer( min_df=2, max_df=0.95, max_features = 20000, 
                              ngram_range = ( 1, 4 ), sublinear_tf = True )
                              
                              
print "Vectorizing..."
                              
vectorizer = vectorizer.fit( np.append( train['review'].values, unlab_train['review'].values  ) )
                              
#Vectorize each review in train dataset
train_data_features = vectorizer.transform( train['review'].values )

#Vectorize each review in test dataset
test_data_features = vectorizer.transform( test['review'].values )


print "Reducing dimension..."


from sklearn.feature_selection.univariate_selection import SelectKBest, chi2

fselect = SelectKBest(chi2 , k=5000)

train_data_features = fselect.fit_transform( train_data_features , train["sentiment"])
test_data_features = fselect.transform( test_data_features )



#==============================================================================
#Search for optimized paramaters using GridSearchCV 
#==============================================================================

print "Finding optimimal alpha for naive bayes..."

param_grid = { 'alpha':[0.0005, 0.001, 0.01, 0.1, 1.0] }

gbr = MultinomialNB()

grid = GridSearchCV( gbr , param_grid,
cv=ShuffleSplit(n=train_data_features.shape[0], n_iter=10, test_size=0.25),
scoring="mean_squared_error",
n_jobs=5 ).fit( train_data_features , train['sentiment'].values  )

model1 = grid.best_estimator_
model1.fit( train_data_features, train["sentiment"] )



print "Finding optimimal parameters for SGD..."

param_grid = { 'loss': ['hinge', 'modified_huber'] , 
               'shuffle' : [True, False] , 
                'random_state':[0, None] }

gbr = SGDClassifier()

grid = GridSearchCV( gbr , param_grid,
cv=ShuffleSplit(n=train_data_features.shape[0], n_iter=10, test_size=0.25),
scoring="mean_squared_error",
n_jobs=8 ).fit( train_data_features , train['sentiment'].values  )


#model2 = SGDClassifier(loss='modified_huber', n_iter=5, random_state=0, shuffle=True)
model2 = grid.best_estimator_
model2.fit( train_data_features, train["sentiment"] )

p1 = model1.predict_proba( test_data_features )[:,1]
p2 = model2.predict_proba( test_data_features )[:,1]

pred_df = pd.read_csv('sampleSubmission.csv')

pred_df['sentiment'] = 1/6*p1 + 5/6*p2

pred_df.to_csv( 'final_output.csv' , index=False )

"""
print 'Creating memory mapped arrays...'
filename = 'dataset.joblib'

joblib.dump(np.asarray( train_data_features , dtype=np.float32, order='F'), filename)
train_data_features = joblib.load(filename, mmap_mode='c')



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

clf = GradientBoostingRegressor( n_estimators=100 )

clf.fit( train_data_features , train['sentiment'].values  )


pred_df = pd.read_csv('sampleSubmission.csv')

pred_df['sentiment'] = clf.predict(  test_data_features )

pred_df.to_csv( 'final_output.csv' , index=False )


#==============================================================================
# Delete memory mapped files
#==============================================================================
for file in os.listdir("./"):
    if ( file.find('dataset') >= 0 ):
        try:        
            os.remove( file )
        except OSError, e:
            print ("Error: %s - %s." % (e.filename,e.strerror))
            
"""        
end = time.time()

print 'Time elapsed ', round( (end - start ) / 60 , 2 ), ' minutes'