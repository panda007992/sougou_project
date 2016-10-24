# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
import numpy
import jieba
import re

label_type = {'age':[1, (1, 2, 3, 4, 5, 6)], 'Gender' : [2, (1, 2)], 'Education' : [3, (1, 2, 3, 4, 5, 6)]}

def generateTrainData(labelName):  
	fr = open('user_tag_query.2W.TRAIN')
	label = []
	m = 1
	trainData = []
	labelIndex = int(label_type[labelName][0])  #得到label的维度
	stopWordSet = []
	fr2 = open('stopWord.txt')
	for line in fr2.readlines():
		line = line.strip()
		stopWordSet.append(line.decode('utf-8'))
	for line in fr.readlines():  #遍历每行	
		line = line.strip().split()
		length = len(line)
		if(length < 5):
			print "length short"
			continue
		if(labelName not in label_type):
			print "labelName not in label_type"
			continue
		if(int(line[labelIndex]) not in label_type[labelName][1]):
			print line[labelIndex]
			continue
		label.append(int(line[labelIndex]))
		tmpData = ""
		for i in range(4, length):  #遍历该用户的所有词条
			tmp = jieba.cut_for_search(line[i])
			tmpLegalData = []
			for j in tmp:  #该词条的分词列表
				if (j not in stopWordSet) and (re.match(r'^[+-]?\d+(.\d*)*$',j) is None):  #过滤掉单个字和纯数字（包括带小数点）
					tmpLegalData.append(j)
			tmpData += " ".join(tmpLegalData)
			tmpData += " "
		tmpData = tmpData.strip()		
		trainData.append(tmpData)
	vectorizer = TfidfVectorizer()
	train_data = vectorizer.fit_transform(trainData)
	#print train_data	
	#print label
	return train_data, label

def generateTestData():
	fr = open('user_tag_query.2W.TEST')
	testData = []
	stopWordSet = []
	fr2 = open('stopWord.txt')
	for line in fr2.readlines():
		line = line.strip()
		stopWordSet.append(line.decode('utf-8'))
	for line in fr.readlines():  #遍历每行	
		line = line.strip().split()
		length = len(line)
		tmpData = ""
		for i in range(1, length):  #遍历该用户的所有词条
			tmp = jieba.cut_for_search(line[i])
			tmpLegalData = []
			for j in tmp:  #该词条的分词列表
				if (j not in stopWordSet) and (re.match(r'^[+-]?\d+(.\d*)*$',j) is None):  #过滤掉单个字和纯数字（包括带小数点）
					tmpLegalData.append(j)
			tmpData += " ".join(tmpLegalData)
			tmpData += " "
		tmpData = tmpData.strip()		
		testData.append(tmpData)
	vectorizer = TfidfVectorizer()
	test_data = vectorizer.fit_transform(testData)
	return test_data

def cvTest(data,label):
	clf = MultinomialNB(alpha = 0.01)
	scores = cross_validation.cross_val_score(clf,data,label,cv = 5)
	return scores

def saveTrainData(data,label):
	import pickle
	fr1 = open('trainData.txt','w')
	fr2 = open('label.txt','w')
	pickle.dump(data,fr1)
	pickle.dump(label,fr2)

def loadTrainData():
	import pickle
	fr1 = open('trainData.txt','r')
	fr2 = open('label.txt','r')
	trainData = pickle.load(fr1)
	label = pickle.load(fr2)
	return trainData,label

def saveTestData(data):
	import pickle
	fr = open('testData.txt','w')
	pickle.dump(data,fr)

def loadTestData():
	import pickle
	fr = open('testData.txt','r')
	testData = pickle.load(fr)
	return testData

if __name__ == '__main__':
	#data,label = generateTranData('age')
	#saveTrainData(data,label) 
	#scores = cvTest(data,label)
	#print scores
	test_data = generateTestData()
	saveTestData(test_data)
	'''
	trainData = data[:16000]
	trainLabel = label[:16000]
	testData = data[16000:]
	testLabel = label[16000:]
	clf = MultinomialNB(alpha=0.01)
	clf.fit(trainData, numpy.asarray(trainLabel))
	pred = clf.predict(testData)
	print pred
	m_precision = metrics.precision_score(testLabel, pred,average = 'samples')
	m_recall = metrics.recall_score(testLabel, pred,average = 'samples')
	print m_precision
	print m_recall
	'''



