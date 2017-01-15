#!/usr/bin/env python

import os
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords # Import the stop word list
# Import BeautifulSoup into your workspace
from bs4 import BeautifulSoup 
np.set_printoptions(threshold='nan')


if __name__ == '__main__':
    train = pd.read_json(os.path.join(os.path.dirname(__file__), 'data', '/home/rajababu/Desktop/reddit project/dummy_DB_clean.json'))
print train.shape

print train.columns.values               
print 'The first review is:'
print train["body"][0]

#raw_input("Press Enter to continue...")
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string, and 
    # the output is a single string (
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words )) 


print "shape"
print train.shape
clean_review = review_to_words( train["body"][0] )
print "Clean review 1...................................................................................."
print clean_review
    # Get the number of reviews based on the dataframe column size
num_reviews = train["body"].size
print num_reviews
print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []
for i in range( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )                                                                    
    clean_train_reviews.append( review_to_words( train["body"][i] ))
     # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.

vectorizer = CountVectorizer(analyzer = "word",   \
                             
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

   # fit_transform() does two functions: First, it fits the model
   # and learns the vocabulary; second, it transforms our training data
   # into feature vectors. The input to fit_transform should be a list of 
   # strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)
   # Numpy arrays are easy to work with, so convert the result to an 
   # array
#train_data_features = train_data_features.toarray()
print train_data_features

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
print vocab
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print count, tag