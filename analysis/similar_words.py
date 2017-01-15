# -*- coding: utf-8 -*-

import sys, getopt
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
import regex
from nltk.corpus import stopwords
import json
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
#from pprint import pprint
from math import *
from gensim.models import Word2Vec


# Function to convert a raw comment to a string of words
# The input is a single string and 
# the output is a single string 
# This function filters out non-Latin characters,
# HTML formatting, stop words from English language,
def clean(raw_comment):
    # raw_comment = train[index]["body"] 
    # Remove HTML
    # comment_text = BeautifulSoup(raw_comment).get_text()
    raw_comment = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', raw_comment, flags=re.MULTILINE) 

    # Remove non-letters        
    latin_only = regex.sub(ur'[^\p{Latin}]', u' ', raw_comment)
    letters_only = re.sub(r'\s+', ' ', latin_only) 

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  

    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]   

    # Join the words back into one string separated by space, 
    # and return the result.
    return ( " ".join( meaningful_words ))

# Creates a training set from input file
def create_ts(input_file):
	print "Reading json file..."
	dicts = []
	f = open(input_file)
	count = 0
	for line in iter(f):
	    if count%100000==0:
	        print "line #",count
	    dicts.append(json.loads(line))
	    count = count + 1

	print "loading json dicts..."
	train = json.loads(json.dumps(dicts))
	f.close()
	print ('Size of the dataset was {:d} samples'.format(len(train)))
	return train

# Counts the number of occurence of a specific word
def count(input_file):
	f = open(input_file)
	SUM = 0
	for line in iter(f):
		count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("fuck"), line))
		SUM = SUM + count
	return SUM


# Learns a vocabulary dictionary of all tokens and
# return vocabulary with the number of word occurances
def fit_data():
	# Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000) 

	train_data_features = vectorizer.fit(clean_comments)
	# train_data_features = train_data_features.toarray()
	# print train_data_features
	# vocab = vectorizer.get_feature_names()
	# print len(vectorizer.vocabulary_)
	# dist = np.sum(train_data_features, axis=0)
	# print dist
	return vectorizer
 


if __name__ == '__channelexec__':
	# fileid = pickle.loads(channel.receive())
    # tagger = pickle.loads(channel.receive())
    # corpus_name = channel.receive()
    # corpus = getattr(nltk.corpus, corpus_name)
    while 1:
		input_file = "/media/tiny/" + pickle.loads(channel.receive())
		train = create_ts(input_file)
		comments = []
		clean_comments = []
		for entry in train:
			comments.append(entry["body"])

		#channel.send(len(comments))
		for comment in comments:
		 	clean_comments.append(clean(comment).split())

		cc = Word2Vec(clean_comments)
		tmp = cc.most_similar('new', topn=10)
		channel.send(pickle.dumps(tmp))
