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
import string
from sklearn.cluster import MiniBatchKMeans
import time
from sklearn.decomposition import TruncatedSVD




def clean(raw_comment):
	'''
	Function to convert a raw comment to a string of words
	The input is a single string and 
	the output is a single string 
	This function filters out non-Latin characters,
	HTML formatting, stop words from English language,
	'''
	if raw_comment=="deleted": return ""
	else: 
		# remove non-ascii characters
		raw_comment = filter(lambda x: x in string.printable, raw_comment)

		# remove urls
		raw_comment = re.sub(r"http\S+", "", raw_comment) #re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', raw_comment, flags=re.MULTILINE) 

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

def create_ts(input_file):
	'''
	Creates a training set from input file
	'''
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

def count(input_file):
	'''
	Counts the number of occurence of a specific word
	'''
	f = open(input_file)
	SUM = 0
	for line in iter(f):
		count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape("fuck"), line))
		SUM = SUM + count
	return SUM

def learn_vocabulary(dataset):
	'''
	Learns a vocabulary dictionary of all tokens and
	return vocabulary with the number of word occurances
	'''
	# Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000) 

	train_data_features = vectorizer.fit_transform(dataset)
	train_data_features = train_data_features.toarray()
	# print train_data_features
	vocab = vectorizer.get_feature_names()
	# print len(vectorizer.vocabulary_)
	dist = np.sum(train_data_features, axis=0)
	# print dist
	return zip(vocab,dist)
 
def fit_data(dataset, new_vocabulary):
	'''
	Uses joint vocabulary to fit and transform 
	orginal dataset
	'''
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             preprocessor = None, \
	                             stop_words = None,   \
	                             max_features = 5000, \
	                             vocabulary = new_vocabulary) 
	vectorizer.vocabulary_ = new_vocabulary
	trasfromed_comments = vectorizer.fit_transform(dataset)
	# train_data_features = train_data_features.toarray()
	return trasfromed_comments.toarray()

def predict(dataset, cluster_centers):
	'''
	Uses MiniBatchKMeans to predict cluster
	for every entry in the dataset
	'''
	kmeans = MiniBatchKMeans(n_clusters=n_clusters)
	# array of cluster labels for each data sample
	kmeans.cluster_centers_ = cluster_centers
	labels = kmeans.predict(dataset)
	return labels

def find_centers(dataset, n_clusters):
	'''
	Uses MiniBatchKMeans to predict cluster
	for every entry in the dataset
	'''
	kmeans = MiniBatchKMeans(n_clusters=n_clusters)
	labels = kmeans.fit(dataset)
	return kmeans.cluster_centers_

def new_centers(dataset, n_clusters, n_features, centers):
	'''
	Calculates new centers for clusters 
	by computing the average over all samples
	that belong to the cluster
	'''

	labels = predict(dataset, centers)
	tmp = []
	# calculate new centers for each cluster
	new_centers = np.zeros((n_clusters,n_features))
	for i in range(n_clusters):
		sample_nums = np.where(labels==i)[0]
		cluster_samples = []
		for index in sample_nums:
			cluster_samples.append(dataset[index])
		# print "cluster #%d has %d samples" % (i,len(cluster_samples))
		tmp.append((i,len(cluster_samples)))
		for k in range(n_features):
			for j in range(len(cluster_samples)):
				new_centers[i][k] += cluster_samples[j][k]
			if (len(cluster_samples) == 0):
				new_centers[i][k] = centers[i][k]
			else:
				new_centers[i][k] = new_centers[i][k]/len(cluster_samples)

	return new_centers


if __name__ == '__channelexec__':

    while 1:

		input_file = pickle.loads(channel.receive())

		info_channel = channel.receive()
		train = create_ts("/media/tiny/" + input_file)
		comments = []
		clean_comments = []
		for entry in train:
			comments.append(entry["body"])
		for comment in comments:
			if comment!=u'deleted':
		 		clean_comments.append(clean(comment))
		info_channel.send(pickle.dumps(learn_vocabulary(clean_comments)))
		# channel.send(pickle.dumps(clean_comments[:1]))


		new_voc = pickle.loads(channel.receive())
		centers = pickle.loads(channel.receive())

		transformed_set = fit_data(clean_comments, new_voc)

		n_clusters = centers.shape[0]
		n_features = centers.shape[1]

		svd = TruncatedSVD(n_components=300)
		truncated_set = svd.fit_transform(transformed_set)
		# n_features = truncated_set.shape[1]
		# print truncated_set.shape

		# channel.send(pickle.dumps(new_centers(truncated_set,n_clusters,n_features, centers)))
		# info_channel = channel.receive()
		info_channel.send(pickle.dumps(new_centers(truncated_set,n_clusters,n_features, centers)))
		# info_channel.receive()
		# info_channel = channel.receive()
		# # info_channel.send(info_channel.receive()+1)
		for i in range(5):
			centers = pickle.loads(channel.receive())
			info_channel.send(pickle.dumps(new_centers(truncated_set, n_clusters, n_features, centers)))




