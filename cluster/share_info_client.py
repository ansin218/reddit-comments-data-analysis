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

if __name__ == '__channelexec__':

	while 1:
		info_channel = channel.receive()
		info_channel.send(1)
		for i in range(5):
			x = channel.receive()
			info_channel.send(x+1+i)

		
