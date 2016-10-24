# -*- coding: utf-8 -*-
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
import numpy
import jieba
import re

label_type = {'age':[1, (0, 1, 2, 3, 4, 5, 6)], 'Gender' : [2, (1, 2)], 'Education' : [3, (0, 1, 2, 3, 4, 5, 6)]}

def generateData(labelName):  
	fr = open('../datas/user_tag_query.2W.TRAIN')
	label = []
	m = 1
	trainData = []
	labelIndex = int(label_type[labelName][0])
	print labelIndex
	for line in fr.readlines()[0:20000]:  #遍历每行	
		line = line.strip('\n').split()
		length = len(line)
		if(length < 5):
			print "length short"
			continue
		if(labelName not in label_type):
			print "labelName not in label_type"
			continue
		if(int(line[labelIndex]) not in label_type[labelName][1]):
			print line[1]
			print "int(line[labelIndex]) not in label_type[labelName][1]"
			continue
		label.append(int(line[labelIndex]))
		tmpData = ""
		for i in range(4, length):  #遍历该用户的所有词条
			tmp = jieba.cut_for_search(line[i])
			for j in tmp:  #该词条的分词列表
				if len(j) > 1 and (re.match(r'^[+-]?\d+(.\d*)*$',j) is None):  #过滤掉单个字和纯数字（包括带小数点）
					tmpData = " ".join(tmp)
		
		trainData.append(tmpData)
	vectorizer = HashingVectorizer(non_negative=True)
	train_data = vectorizer.fit_transform(trainData)
	print train_data	
	print label
	return train_data, label

def train_and_test_data(data):
	filesize = int(0.7 * len(data))
	vectorizer = HashingVectorizer(tokenizer=comma_tokenizer, non_negative=True)
	train_data = [each[0] for each in data[:filesize]]
	train_target = [each[1] for each in data[:filesize]]
	test_data = [each[0] for each in data[filesize:]]
	test_target = [each[1] for each in data[filesize:]]
	return train_data, train_target, test_data, test_target


#data = generateData("Gender")
#train_data, train_target, test_data, test_target = train_and_test_data(data)
#
#nbc = Pipeline([('vect', TfidfVectorizer()),('clf', MultinomialNB(alpha=1.0)),])
#nbc.fit(train_data, train_target)    #训练我们的多项式模型贝叶斯分类器
#predict = nbc_6.predict(test_data)  #在测试集上预测结果
#count = 0                                      #统计预测正确的结果个数
#for left , right in zip(predict, test_target):
#	if left == right:
#		count += 1
#		print(count/len(test_target))
#
#nbc1= Pipeline([('vect', TfidfVectorizer()),('clf', BernoulliNB(alpha=0.1)),])
#predict = nbc_1.predict(test_data)  #在测试集上预测结果
#count = 0                                      #统计预测正确的结果个数
#for left , right in zip(predict, test_target):
#	if left == right:
#		count += 1
#		print(count/len(test_target))
#
if __name__ == '__main__':
	data,label = generateData("age")
	trainData = data[:16000]
	trainLabel = label[:16000]
	testData = data[16000:]
	testLabel = label[16000:]
	clf = MultinomialNB(alpha=0.01)
	clf.fit(trainData, numpy.asarray(trainLabel))
	pred = clf.predict(testData)
	print pred
	m_precision = metrics.precision_score(testLabel, pred, average='samples')
	m_recall = metrics.recall_score(testLabel, pred, average='samples')
	print m_precision
	print m_recall




