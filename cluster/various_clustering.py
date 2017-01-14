import execnet
import nltk
import clusterC
import os
import cPickle as pickle
from collections import Counter
import numpy as np
import time
import help_functions as hp
from sklearn.cluster import MiniBatchKMeans, DBSCAN, KMeans
from scipy.spatial.distance import cdist, pdist
from sklearn import cluster, metrics
from sklearn.feature_extraction.text import CountVectorizer

from time import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


# tss = pickle.load(open("/media/small/pickled_tss","r+"))
# bss = pickle.load(open("/media/small/pickled_bss","r+"))
# print bss/tss

# truncated ds
# X = pickle.load(open("/media/small/transformed_dataset","r+"))
# # print X.shape
# # for n_clusters in range(1,30):
# # 	km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++')
# # 	km.fit(X)
# # 	for i in range(n_clusters): 
# # 		print "cluster #", i, "\t", sum(1 for l in km.labels_ if l==i)
# # 	print "############################################################"




# print "trying different k..."
# kMeansVar = [MiniBatchKMeans(n_clusters=k).fit(X) for k in range(1,30)]
# centroids = [K.cluster_centers_ for K in kMeansVar]
# k_euclid = [cdist(X, cent) for cent in centroids]
# dist = [np.min(ke, axis=1) for ke in k_euclid]
# wcss = [sum(d**2) for d in dist]
# tss = sum(pdist(X)**2)/X.shape[0]
# bss = tss - wcss
# pickle.dump(bss, open("/media/small/pickled_bss_1", "w+"))
# pickle.dump(tss, open("/media/small/pickled_tss_1", "w+"))

X = pickle.load(open("/media/jan2015/tmp/X","r+"))
subreddits = pickle.load(open("/media/jan2015/tmp/subreddits","r+"))
# data = pickle.load(open("/media/test/data","r+"))

# for row in data:
# 	row = row.split()
# 	# print len(row)
# print row
# vectorizer = CountVectorizer(analyzer = "word",   \
# 	                             preprocessor = None, \
# 	                             stop_words = None,   \
# 	                             max_features = 2000)

# X = vectorizer.fit_transform(data).todense()
# print vectorizer.vocabulary_
# print len(vectorizer.vocabulary_)
# print X[0]
# DBSCAN

# db = DBSCAN().fit(X)
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print n_clusters_
# for i in range(n_clusters_): 
# 	print "cluster #", i, "\t", sum(1 for l in labels if l==i)
t0 = time()
# kMeansVar = [MiniBatchKMeans(n_clusters=k).fit(X) for k in range(1,100)]
# print("done in %0.3fs" % (time() - t0))
# labels = [km.labels_ for km in kMeansVar]
# silhouette_scores = [metrics.silhouette_score(X, labels_, sample_size=1000) for labels_ in labels]
# print silhouette_scores

km = MiniBatchKMeans(n_clusters=20, init='k-means++')
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))

for i in range(20): 
	print "cluster #", i, "\t", sum(1 for l in km.labels_ if l==i)
	count = 0
	for subreddit,label in zip(subreddits,km.labels_):
		if label==i:
			if count<50:
				print subreddit
				count += 1
	print "######################################################"