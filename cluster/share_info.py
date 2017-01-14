import execnet
import nltk
import share_info_client
import os
import cPickle as pickle
from collections import Counter
import numpy as np
import time
import help_functions as hp


DIR = "/media/tiny"

HOSTS = {
	'10.155.208.100': 5,
	'10.155.208.101': 4}
	# '10.155.208.111': 8, 
	# '10.155.208.113': 8, 
	# '10.155.208.75': 8}

NICE = 20

channels = []
gateways = {}
for host,count in HOSTS.items():
	print 'opening %d gateways at %s' % (count, host)

	for i in range(count):
		gw = execnet.makegateway('ssh=%s//nice=%d' % (host, NICE))
		channel = gw.remote_exec(share_info_client)
		gateways[channel] = gw
		channels.append(channel)

count = 0
chan = 0


# for root, dirs, fileids in os.walk(DIR):
#     for fileid in fileids:
#     	if fileid[0] != "p":
# 		    print 'sending %s to channel %d' % (fileid, chan)
# 		    channels[chan].send(0)
# 		    channels[chan].send(pickle.dumps(fileid))
# 		    count += 1
# 		    # alternate channels
# 		    chan += 1
# 		    if chan >= len(channels): chan = 0

# print "Done sending to channels. Executing script..."
 
multi = execnet.MultiChannel(channels)

queue = multi.make_receive_queue()
clean_comments = []
# vocabs = []
# for i in range(count):
# 	channel, response = queue.get()
# 	comments, vocab = pickle.loads(response)
# 	vocabs.append([k for k,v in vocab])
# 	clean_comments.extend(comments)

# merge all vocabularies summing up number of word occurances
# joint_vocab = sum((Counter(dict(x)) for x in vocabs),Counter())

# intersect all vocabularies selecting only common words 
# joint_vocab = list(set.intersection(*(set(d) for d in vocabs)))

#filter out infrequent words
# joint_vocab = list({k for k,v in joint_vocab.iteritems() if v > 10})

# print joint_vocab
# print "current joint vocabulary consists of %d entries" % len(joint_vocab)


# parallel k-means

# channels for sending info
info_channels = []
send_channels = []
new_centers = []
# n_clusters = 10
# n_features = 300#len(joint_vocab)

# print centers

# print clean_comments
# # clean_comments = pickle.load(open("/media/tiny/p_xzbcu", "rb"))
# transformed_set = hp.fit_data(clean_comments, joint_vocab)[:n_clusters]
# # centers = hp.find_centers(transformed_set,n_clusters)
# centers = np.random.choice([0, 1], size=[n_clusters,n_features]) 
# print centers.shape

count = 0
chan = 0
for root, dirs, fileids in os.walk(DIR):
	for fileid in fileids:
		if fileid[0] != "p":
			
			print 'sending %s to channel %d' % (fileid, chan)
			# channels[chan].send(1)
			# channels[chan].send(pickle.dumps(fileid))
			info_channel = gateways[channels[chan]].newchannel()
			info_channels.append(info_channel)
			
			# channels[chan].send(pickle.dumps(joint_vocab))
			# channels[chan].send(pickle.dumps(centers))

			channels[chan].send(info_channel)
			# 
			# channels[chan].send(1)
			count += 1
			# alternate channels
			chan += 1
			if chan >= len(channels): chan = 0
x = 0



for i in range(count):
	# print info_channels[i]
	x += info_channels[i].receive()

for k in range(5):
	chan = 0
	for i in range(count):
		# send_channel = gateways[channels[chan]].newchannel()
		# send_channels.append(send_channel)
		channels[chan].send(x)
		# send_channel.send(x)
		chan += 1
		if chan >= len(channels): chan = 0

	for info_channel in info_channels:
		print info_channel.receive()

# multi = execnet.MultiChannel(new_channels)
# queue = multi.make_receive_queue()

# for i in range(count):
# 	print "response #", i
# 	channel, response = queue.get()
# 	print channel, pickle.loads(response)
	# print pickle.loads(response)

