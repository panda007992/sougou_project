# -*- coding: <utf-8> -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
import numpy as np


news = fetch_20newsgroups(subset='all')
print type(news)
print "key:", news.keys()
print "Type:", type(news.data), type(news.target), type(news.target_names)
print "targetname:", news.target_names
print "length data:", len(news.data)
print "news target:", len(news.target)
print "news data:", news.data[0]
print "news target:", news.target[0], "target name:", news.target_names[news.target[0]]

#data pre-process
SPLIT_PERC = 0.75
split_size = int(len(news.data)*SPLIT_PERC)
X_train = news.data[:split_size]
X_test = news.data[split_size:]
Y_train = news.target[:split_size]
Y_test = news.target[split_size:]

#nbc means naive bayes classifier
nbc_1 = Pipeline([('vect', CountVectorizer()),('clf', MultinomialNB()),])
nbc_2 = Pipeline([('vect', HashingVectorizer(non_negative=True)),('clf', MultinomialNB()),])
nbc_3 = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB()),])
nbcs = [nbc_1, nbc_2, nbc_3]


def evaluate_cross_validation(clf, X, y, K):
	# create a k-fold croos validation iterator of k=5 folds
	cv = KFold(len(y), K, shuffle=True, random_state=0)
	# by default the score used is the one returned by score method of the estimator (accuracy)
	scores = cross_val_score(clf, X, y, cv=cv)
	print scores
	print ("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores))


for nbc in nbcs:
	    evaluate_cross_validation(nbc, X_train, Y_train, 5)



