import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import sentiwordnet as swn
import numpy as np

# This code Identify the sentiments which score highly
# We separated and scored the dataset into the top quartile by the sentiments:
# Positive, Negative, Objective, Subjective
# We are using sentiwordnet library for senti words.

def get_scores(x):
    return list(swn.senti_synsets(x))

def get_positive_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].pos_score()
    return 0

def get_negative_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].neg_score()
    return 0

def get_objective_score(sentiments):
    if len(sentiments) > 0:
        return sentiments[0].obj_score()
    return 0

sql_conn = sqlite3.connect('database.sqlite')

df = pd.read_sql("SELECT score, body FROM May2015 WHERE LENGTH(body) > 30 AND LENGTH(body) < 300 LIMIT 800000", sql_conn)

keywords = ['Positive', 'Negative', 'Objective', 'Subjective']

content_summary = pd.DataFrame()

pos_content = []
neg_content = []
obj_content = []

# get the average score for all words in the comments
for string in df['body'].values:
    strings = string.split(" ")
    string_scores = list(map(lambda x: get_scores(x), strings))
    pos_scores = list(map(lambda x: get_positive_score(x), string_scores))
    neg_scores = list(map(lambda x: get_negative_score(x), string_scores))
    obj_scores = list(map(lambda x: get_objective_score(x), string_scores))

    pos_content.append(np.mean(pos_scores))
    neg_content.append(np.mean(neg_scores))
    obj_content.append(np.mean(obj_scores))

df['Positive'] = pos_content
df['Negative'] = neg_content
df['Objective'] = obj_content

#print(df)

# get the above average and (top quartile or top 3/8) comments for each sentiment
pos_mean = np.mean(df['Positive'].values)
pos_content = df[df.Positive.apply(lambda x: x > pos_mean * 2.5)]
content_summary['Positive'] = pos_content.describe().score

neg_mean = np.mean(df['Negative'].values)
neg_content = df[df.Negative.apply(lambda x: x > neg_mean * 2.5)]
content_summary['Negative'] = neg_content.describe().score

obj_mean = np.mean(df['Objective'].values)
obj_content = df[df.Objective.apply(lambda x: x > obj_mean * 1.5)]
content_summary['Objective'] = obj_content.describe().score

subj_content = df[df.Objective.apply(lambda x: x < obj_mean * 0.5)]
content_summary['Subjective'] = subj_content.describe().score

keys = keywords

content_summary = content_summary.transpose() 
print " printing content summary"
print content_summary

# Setting the bars
pos = list(range(len(content_summary['count'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

clrs = []
for v in content_summary['mean'].values:
    if v < 6:
        clrs.append('#FFC1C1')
    elif v < 6.5:
        clrs.append('#F08080')
    elif v < 7.5:
        clrs.append('#EE6363')
    else:
        clrs.append('r')

plt.bar(pos,
        content_summary['count'],
        width,
        alpha=0.5,
        # with color
        color=clrs,
        label=keys)

# Setting y axis label
ax.set_ylabel('Number of comments')

# Setting the chart's title
ax.set_title('sentiments with highest ranks')

# Setting the position of the x ticks
ax.set_xticks([p + 0.5 * width for p in pos])

# Setting the labels for the x ticks
ax.set_xticklabels(keys)

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)-width, max(pos)+width*4)
plt.ylim([0, max(content_summary['count'])* 1.5])

rects = ax.patches

for ii,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% ("Score {0:.2f}".format(content_summary['mean'][ii])),
                 ha='center', va='bottom')

plt.grid()

plt.savefig("sentiments_vs_rank_new.png")
