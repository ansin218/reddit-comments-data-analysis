import execnet
import nltk
import most_popular_subreddits
import os
import cPickle as pickle
from collections import Counter
import numpy as np
import time
import help_functions as hp

start_time = time.time()
DIR = "/media/jan2015"

HOSTS = {
	'10.155.208.100': 8,
	'10.155.208.113': 8,
	'10.155.208.60': 8, 
	'10.155.208.76': 8, 
	'10.155.208.75': 8}

NICE = 20

channels = []

# channels for sending info
info_channels = []

gateways = {}

for host,count in HOSTS.items():
	print 'opening %d gateways at %s' % (count, host)

	for i in range(count):
		gw = execnet.makegateway('ssh=%s//nice=%d' % (host, NICE))
		channel = gw.remote_exec(most_popular_subreddits)
		gateways[channel] = gw	
		channels.append(channel)

count = 0
chan = 0


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
 
multi = execnet.MultiChannel(channels)
# info_multi = execnet.MultiChannel(info_channels)

queue = multi.make_receive_queue()
users_global = []
subreddits_global = []
# info_queue = info_multi.make_receive_queue()
# collections = {}
for i in range(count):
	print i
	users = pickle.loads(info_channels[i].receive())
	print len(users)
	# users = pickle.loads(info_channels[i].receive())
	# collections.update(collection)
	users_global = list(set(users + users_global))
	print len(users_global)
	print "#######################################"

for i in range(count):
	print i
	subreddits = pickle.loads(info_channels[i].receive())
	print len(subreddits)
	# users = pickle.loads(info_channels[i].receive())
	# collections.update(collection)
	# users_global = list(set(users + users_global))
	subreddits_global = list(set(subreddits + subreddits_global))
	print len(subreddits_global)
	print "#######################################"
# # result = sum((Counter(dict(x)) for x in collections),Counter())
# result = sorted(collections.items(), key=lambda x:x[1])[-5000:]
# for k,v in result:
# 	print k,v
# print len(result)
print len(users_global),"users_global"
print len(subreddits_global), "subreddits_global"
print "elapsed time: ", time.time() - start_time
# print len(users_global),"users_global"
# pickle.dump(result, open("/media/fin/pickled_top_subreddits","w+"))

