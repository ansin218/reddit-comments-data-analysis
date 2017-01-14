# coding=UTF-8
import nltk
from nltk.corpus import brown
import re
import regex
import operator
import json
import cPickle as pickle
import sys, getopt

import numpy as np
from nltk.corpus import stopwords

import string
import time


nltk.data.path.append("/media/nltk")

brown_train = brown.tagged_sents(categories='news')
regexp_tagger = nltk.RegexpTagger(
    [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'),
     (r'(-|:|;)$', ':'),
     (r'\'*$', 'MD'),
     (r'(The|the|A|a|An|an)$', 'AT'),
     (r'.*able$', 'JJ'),
     (r'^[A-Z].*$', 'NNP'),
     (r'.*ness$', 'NN'),
     (r'.*ly$', 'RB'),
     (r'.*s$', 'NNS'),
     (r'.*ing$', 'VBG'),
     (r'.*ed$', 'VBD'),
     (r'.*', 'NN')
])
unigram_tagger = nltk.UnigramTagger(brown_train, backoff=regexp_tagger)
bigram_tagger = nltk.BigramTagger(brown_train, backoff=unigram_tagger)

cfg = {}
cfg["NNP+NNP"] = "NNP"
cfg["NN+NN"] = "NNI"
cfg["NNI+NN"] = "NNI"
cfg["JJ+JJ"] = "JJ"
cfg["JJ+NN"] = "NNI"


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
        # letters_only = re.sub("[^a-zA-Z]", " ", raw_comment) 
        
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


class NPExtractor(object):

    def __init__(self, sentence):
        self.sentence = sentence

    # Split the sentence into singlw words/tokens
    def tokenize_sentence(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    # Normalize brown corpus' tags ("NN", "NN-PL", "NNS" > "NN")
    def normalize_tags(self, tagged):
        n_tagged = []
        for t in tagged:
            if t[1] == "NP-TL" or t[1] == "NP":
                n_tagged.append((t[0], "NNP"))
                continue
            if t[1].endswith("-TL"):
                n_tagged.append((t[0], t[1][:-3]))
                continue
            if t[1].endswith("S"):
                n_tagged.append((t[0], t[1][:-1]))
                continue
            n_tagged.append((t[0], t[1]))
        return n_tagged

    # Extract the main topics from the sentence
    def extract(self):

        tokens = self.tokenize_sentence(self.sentence)
        tags = self.normalize_tags(bigram_tagger.tag(tokens))

        merge = True
        while merge:
            merge = False
            for x in range(0, len(tags) - 1):
                t1 = tags[x]
                t2 = tags[x + 1]
                key = "%s+%s" % (t1[1], t2[1])
                value = cfg.get(key, '')
                if value:
                    merge = True
                    tags.pop(x)
                    tags.pop(x)
                    match = "%s %s" % (t1[0], t2[0])
                    pos = value
                    tags.insert(x, (match, pos))
                    break

        matches = []
        for t in tags:
            if t[1] == "NNP" or t[1] == "NNI":
            #if t[1] == "NNP" or t[1] == "NNI" or t[1] == "NN":
                matches.append(t[0])
        return matches

if __name__ == '__channelexec__':

    while 1:

        input_file = pickle.loads(channel.receive())
        info_channel = channel.receive()
        train = create_ts("/media/jan2015/tmp/" + input_file)
        comments = []
        clean_comments = []
        subreddits = []
        
        # load the list of top subreddits
        # subreddits = [k for k,v in pickle.load(open("/media/fin/pickled_top_subreddits","r+"))]
        for entry in train:
            if entry["subreddit"] not in subreddits:
                subreddits.append(entry["subreddit"])


        # dictionary with key=subreddit and value = list of clean comment
        collection = {}
        count = 0
        for subreddit in subreddits:
            collection[subreddit] = list()
            for entry in train:
                if entry["subreddit"] == subreddit:
                    count += 1
                    collection[subreddit].append(clean(entry["body"]))

        # in topics: key=subreddit, value = string of topics (verb,adjective pairs)
        topics = {}
        for (subreddit,comments) in collection.iteritems(): 
            tmp = []
            for sentence in comments:
                np_extractor = NPExtractor(sentence)
                result = np_extractor.extract()
                tmp.extend(result)
            topics[subreddit] = tmp
        
        # changed collection: key=subreddit, value=dictionary of words and their frequencies
        result = []
        for (subreddit,topic) in topics.iteritems():
            # print topic
            count = {}
            tmp = []
            for word in [r.split() for r in topic]:
                tmp.extend(word)
            for word in tmp:
                count[word] = tmp.count(word)
            collection[subreddit] = count


        info_channel.send(pickle.dumps(collection))

        
