# coding=UTF-8
import nltk
from nltk.corpus import brown
import re
import regex
import operator
import json
import cPickle as pickle
import sys, getopt
from collections import Counter
import numpy as np
from nltk.corpus import stopwords

import string
import time

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

if __name__ == '__channelexec__':

    while 1:

        input_file = pickle.loads(channel.receive())
        info_channel = channel.receive()
        train = create_ts("/media/small/" + input_file)
        subreddits = []

        for entry in train:
            subreddit = entry["subreddit"]
            if subreddit not in subreddits:
                subreddits.append(subreddit)
        # print "# of subreddits: ", len(subreddits)

        collection = {}
        count = 0
        for subreddit in subreddits:
            collection[subreddit] = 0
            for entry in train:
                if entry["subreddit"] == subreddit:
                    collection[subreddit] += 1
                    # collection[subreddit].append(clean(entry["body"]))

        info_channel.send(pickle.dumps(sorted(collection.items(), key=lambda x:x[1])))
         