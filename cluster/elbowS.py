import execnet
import elbow
import os
import cPickle as pickle
from collections import Counter
import numpy as np
import time
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

n_clusters = range(1,100)
kMeansVar = []
X = pickle.load(open("/media/jan2015/tmp/X","r+"))

HOSTS = {
	'10.155.208.39': 20,
	'10.155.208.73': 20,
	'10.155.208.76': 20, 
	'10.155.208.45': 20,
	'10.155.208.60': 20}

NICE = 20

channels = []
gateways = {}

for host,count in HOSTS.items():
	print 'opening %d gateways at %s' % (count, host)

	for i in range(count):
		gw = execnet.makegateway('ssh=%s//nice=%d' % (host, NICE))
		channel = gw.remote_exec(elbow)
		gateways[channel] = gw	
		channels.append(channel)

count = 0
chan = 0
t0 = time()
for n in n_clusters:
	print 'sending %s to channel %d' % (n, chan)
	# channels[chan].send(0)
	channels[chan].send(n)
	count += 1
    # alternate channels
	chan += 1
	if chan >= len(channels): chan = 0


multi = execnet.MultiChannel(channels)

queue = multi.make_receive_queue()

for i in range(count):
	print "response #", i
	channel, response = queue.get()
	kMeansVar.append(pickle.loads(response))
print "got results ", time()-t0
t0=time()
centroids = [K.cluster_centers_ for K in kMeansVar]
k_euclid = [cdist(X, cent) for cent in centroids]
dist = [np.min(ke, axis=1) for ke in k_euclid]
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(X)**2)/X.shape[0]
bss = tss - wcss
pickle.dump(bss, open("/media/jan2015/tmp/pickled_bss", "w+"))
pickle.dump(tss, open("/media/jan2015/tmp/pickled_tss", "w+"))
print "done in", time()-t0
print bss/tss*100.0

