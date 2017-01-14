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

from time import time

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


DIR = "/media/jan2015/tmp"

HOSTS = {
	'10.155.208.39': 8,
	'10.155.208.73': 8,
	'10.155.208.76': 8, 
	'10.155.208.45': 8,
	'10.155.208.60': 8}

NICE = 20

channels = []

# channels for sending info
info_channels = []

gateways = {}

for host,count in HOSTS.items():
	print 'opening %d gateways at %s' % (count, host)

	for i in range(count):
		gw = execnet.makegateway('ssh=%s//nice=%d' % (host, NICE))
		channel = gw.remote_exec(clusterC)
		gateways[channel] = gw	
		channels.append(channel)

count = 0
chan = 0

t0 = time()
for root, dirs, fileids in os.walk(DIR):
    for fileid in fileids:
    	if fileid[0] == "x":
			print 'sending %s to channel %d' % (fileid, chan)
			# channels[chan].send(0)
			channels[chan].send(pickle.dumps(fileid))
			info_channel = gateways[channels[chan]].newchannel()
			info_channels.append(info_channel)
			channels[chan].send(info_channels[chan])
			count += 1
		    # alternate channels
			chan += 1
			if chan >= len(channels): chan = 0

print "Done sending to channels. Executing script..."
print("sending done in %f s" % (time() - t0))

t0=time()
multi = execnet.MultiChannel(channels)
# info_multi = execnet.MultiChannel(info_channels)

queue = multi.make_receive_queue()
# info_queue = info_multi.make_receive_queue()

# receive dectionary from each file with key=subreddit, value=dictionary of words and their frequency
collections = []
subreddits = []
for i in range(count):
	print "channel", i
	collection = pickle.loads(info_channels[i].receive())
	for subreddit,count in collection.iteritems():
		if subreddit not in subreddits:
			subreddits.append(subreddit)
	collections.append(collection)

print "got results from clients"
print("done in %f s" % (time() - t0))

pickle.dump(collections, open("/media/jan2015/tmp/collections","w+"))
pickle.dump(subreddits, open("/media/jan2015/tmp/subreddits","w+"))
# subreddits = pickle.load(open("/media/test/subreddits","r+")) 
# collections = pickle.load(open("/media/test/collections","r+")) 

joint_topics = {}
# load top subreddits
# subreddits = [k for k,v in pickle.load(open("/media/fin/pickled_top_subreddits","r+"))]
print "Total number of subreddits: ", len(subreddits)
for s in subreddits:
	joint_topics[s] = {}

# merge dictionaries from each file into general one 
count = 0
for collection in collections:
	for k,v in collection.iteritems():
		if k in subreddits:
			joint_topics[k].update(v)

top_subreddits = {}
print "# of subreddits before thresholding: ", len(joint_topics)
for subreddit,words in joint_topics.iteritems():
	if len(words)>=500:
		top_subreddits[subreddit]=dict(sorted(words.iteritems(), key=lambda (k, v): (-v, k))[:500])

# # joint_topics = {k:joint_topics[k] for k in joint_topics if len(v)>1000}

# # print "# of subreddits after thresholding: ", len(top_subreddits)

# each line of data = words extracted from corresponding subreddit
data = []
for subreddit,words in top_subreddits.iteritems():
	data.append(" ".join(list(words)))
	# data.append(list(words))


# print data[0]
# print type(data[0])
# print "length of data", len(data)
# training dataset consists of all words
# dataset = []
# for k,v in joint_topics.iteritems():
# 	dataset.extend(v)
# print len(dataset)

# print "Learning vocabulary"
# joint_vocab = hp.learn_vocabulary(dataset)
# joint_words = [k for k,v in joint_vocab]
# # joint_vocab = sorted(dict(joint_vocab).items(), key=lambda x:x[1])
# transformed_dataset = np.zeros((len(subreddits),len(joint_words)))
# joint_words = []
# i = 0
# print "Vectorizing dataset"
# for subreddit,topics in top_subreddits.iteritems():
# 	# topics = sorted(topics, key=topics.get)[-5000:]
# 	# print len(topics),type(topics)
# 	tmp = []
# 	tmp.append(' '.join(topics))
# 	transformed_topics = hp.fit_data(tmp, joint_words)
# 	# print transformed_topics.shape
# 	transformed_dataset[i] = np.array(transformed_topics)
# 	i += 1

pickle.dump(data, open("/media/jan2015/tmp/data","w+"))
# # data = pickle.load(open("/media/small/data","r+"))


print "Length before vectorizing:", len(data)
t0 = time()
vectorizer = TfidfVectorizer(analyzer='word', max_df=0.6, max_features=5000)
X = vectorizer.fit_transform(data)
print len(vectorizer.stop_words_)
print ("done in %f s" % (time() - t0))

print "Dimensionality reduction"
t0 = time()
svd = TruncatedSVD(2000)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(X)
print("done in %f s" % (time() - t0))
print "shape of X: ", X.shape

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

pickle.dump(X, open("/media/jan2015/tmp/X","w+"))


# X = pickle.load(open("/media/small/X","r+"))
# print "Clustering"

# km = MiniBatchKMeans(n_clusters=20, init='k-means++')
# t0 = time()
# km.fit(X)
# print("done in %0.3fs" % (time() - t0))

# # pickle.dump(transformed_dataset, open("/media/test/pickled_ds","w+"))

# # spectral = cluster.SpectralClustering()
# # labels = spectral.fit_predict(transformed_dataset)
# # kmeans = MiniBatchKMeans(n_clusters=20)
# # labels =  kmeans.fit_predict(np.array(transformed_dataset))

# # for i in labels:
# # 	print i, list(labels).count(i)

# # tmp = zip(subreddits,labels)

# # for i in labels:
# # 	print i
# # 	print [k for k,v in tmp if v==i]
# # 	print "##########################"

# print "trying different k..."
# n_clusters = [500,1200,2000]
# kMeansVar = [MiniBatchKMeans(n_clusters=k).fit(X) for k in n_clusters]
# centroids = [K.cluster_centers_ for K in kMeansVar]
# k_euclid = [cdist(X, cent) for cent in centroids]
# dist = [np.min(ke, axis=1) for ke in k_euclid]
# wcss = [sum(d**2) for d in dist]
# tss = sum(pdist(X)**2)/X.shape[0]
# bss = tss - wcss
# pickle.dump(bss, open("/media/small/pickled_bss", "w+"))
# pickle.dump(tss, open("/media/small/pickled_tss", "w+"))

# print bss/tss*100.0

# # pickle.dump(joint_topics, open("/media/small/pickled_subreddits_topics","w+"))
# # print joint_topics
# # for i in range(count):
# # 	print "response #", i
# # 	channel, response = queue.get()
# # 	print channel, pickle.loads(response)
# # 	# print pickle.loads(response)

# DBSCAN

# db = DBSCAN(eps=0.3).fit(X)
# print db
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print n_clusters_
# for i in range(n_clusters_): 
# 	print "cluster #", i, "\t", sum(1 for l in labels if l==i)
# 	# for subreddit,label in zip(subreddits,labels):
# 	# 	if label==i:
# 	# 		print subreddit
# 	# print "######################################################"


# for i in range(20): 
# 	print "cluster #", i, "\t", sum(1 for l in km.labels_ if l==i)
# 	# center = np.where(X==km.cluster_centers_[i])
# 	# print center
# 	# print "centroid: ", km.labels_[center[0][0]]
	
# 	# for subreddit,label in zip(subreddits,km.labels_):
# 	# 	count = 0
# 	# 	if label==i:
# 	# 		if count<50:
# 	# 			print subreddit
# 	# 			count += 1
# 	print "######################################################"

# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, db.labels_, sample_size=1000))


