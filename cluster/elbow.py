import execnet
import cPickle as pickle
from collections import Counter
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist, pdist
from sklearn import cluster, metrics
from time import time


if __name__ == '__channelexec__':

    while 1:
		X = pickle.load(open("/media/jan2015/tmp/X","r+"))
		kMeansVar = MiniBatchKMeans(n_clusters=channel.receive()).fit(X)
		channel.send(pickle.dumps(kMeansVar))