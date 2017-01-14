import execnet
import nltk
import script
import os
import cPickle as pickle
from collections import Counter
import numpy as np
import time
import help_functions as hp


DIR = "/media/tiny"

HOSTS = {
	'10.155.208.100': 5,
	'10.155.208.113': 4}
	# '10.155.208.111': 8, 
	# '10.155.208.113': 8, 
	# '10.155.208.75': 8}

NICE = 20

channels = []

# channels for sending info
info_channels = []

gateways = {}

for host,count in HOSTS.items():
	print 'opening %d gateways at %s' % (count, host)

	for i in range(count):
		gw = execnet.makegateway('ssh=%s//nice=%d' % (host, NICE))
		channel = gw.remote_exec(script)
		gateways[channel] = gw	
		channels.append(channel)

count = 0
chan = 0


for root, dirs, fileids in os.walk(DIR):
    for fileid in fileids:
    	if fileid[0] != "p":
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
 
multi = execnet.MultiChannel(channels)
# info_multi = execnet.MultiChannel(info_channels)

queue = multi.make_receive_queue()
# info_queue = info_multi.make_receive_queue()

clean_comments = []
vocabs = []
for i in range(count):
	vocab = pickle.loads(info_channels[i].receive())
	vocabs.append([k for k,v in vocab])
	# clean_comments.extend(comments)

# merge all vocabularies summing up number of word occurances
# joint_vocab = sum((Counter(dict(x)) for x in vocabs),Counter())

# intersect all vocabularies selecting only common words 
joint_vocab = list(set.intersection(*(set(d) for d in vocabs)))

#filter out infrequent words
# joint_vocab = list({k for k,v in joint_vocab.iteritems() if v > 10})

# print joint_vocab
print "current joint vocabulary consists of %d entries" % len(joint_vocab)


# parallel k-means



n_clusters = 10
n_features = 300#len(joint_vocab)

centers = np.random.choice([0, 1], size=[n_clusters,n_features]) 
print centers.shape

new_centers = np.zeros((n_clusters,n_features))

count = 0
chan = 0
for root, dirs, fileids in os.walk(DIR):
    for fileid in fileids:
    	if fileid[0] != "p":
			channels[chan].send(pickle.dumps(joint_vocab))
			channels[chan].send(pickle.dumps(centers))
			count += 1
			# alternate channels
			chan += 1
			if chan >= len(channels): chan = 0

for i in range(count):
	# print info_channels[i]
	new_centers = np.add(new_centers, pickle.loads(info_channels[i].receive()))

for k in range(5):
	chan = 0
	for i in range(count):
		channels[chan].send(pickle.dumps(new_centers/float(count)))
		# print pickle.loads(info_channels[i].receive())
		chan += 1
		if chan >= len(channels): chan = 0

	for i in range(count):
		print k
		print pickle.loads(info_channels[i].receive())

# multi = execnet.MultiChannel(new_channels)
# queue = multi.make_receive_queue()

# for i in range(count):
# 	print "response #", i
# 	channel, response = queue.get()
# 	print channel, pickle.loads(response)
	# print pickle.loads(response)

