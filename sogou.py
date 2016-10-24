# -*- coding:utf-8 -*-

import numpy
import jieba
import re

def generateData(labelDim):  
	fr1 = open('user_tag_query.2W.TRAIN')
	label = []
	m = 1
	trainSet = []
	trainData = []
	stopWordSet = []
	fr2 = open('stopWord.txt')
	for line in fr2.readlines():
		line = line.strip()
		stopWordSet.append(line.decode('utf-8'))
	stopWordSet = set(stopWordSet)
	for line in fr1.readlines():  #遍历每行	
		print m
		line = line.strip().split()
		n = len(line)
		label.append(int(line[labelDim]))
		tmpData = []
		for i in range(4,n):  #遍历该用户的所有词条
			#tmp = line[i].decode('GBK')	
			tmp = line[i]
			tmp = jieba.cut_for_search(tmp)
			for j in tmp:  #该词条的分词列表
				if (j not in stopWordSet) and (re.match(r'^[+-]?\d+(.\d*)*$',j) is None):  #过滤掉单个字和纯数字（包括带小数点）
					trainSet.append(j)
					tmpData.append(j)
		trainData.append(tmpData)
		m += 1
	trainSet = set(trainSet)
	return trainSet,trainData,label
	
def loadData():  #导入词典 分词矩阵，label
	fr1 = open('trainDic.txt','r')
	fr2 = open('trainData.txt','r')
	fr3 = open('label.txt','r')
	trainDic = []
	trainData = []
	label = []
	for line in fr1.readlines():
		line = line.strip()
		trainDic.append(line)
	for line in fr2.readlines():
		line = line.strip().split()
		tmpData = []
		for i in line:
			tmpData.append(i)
		trainData.append(tmpData)
	for line in fr3.readlines():
		line = line.strip()
		label.append(int(line))
	return trainDic,trainData,label
	
def wordFrequency(trainData): #计算词频
	wordFreq = {}
	m = 1
	for line in trainData:
		#print m
		for i in line:
			if not wordFreq.has_key(i):
				wordFreq[i] = 1
			else:
				wordFreq[i] += 1 
		m += 1
	return wordFreq

def saveData(trainDic,trainData,wordFreq,label):  #将生成的词典 分词样本 词频 label离线保存
	fr1 = open('trainDic.txt','w')
	fr2 = open('trainData.txt','w')
	fr3 = open('wordFreq.txt','w')
	fr4 = open('label.txt','w')
	for key in trainDic:
		print >> fr1,key.encode('utf-8')
	for i in range(len(trainData)):
		for j in range(len(trainData[i])):
			print >> fr2,trainData[i][j].encode('utf-8'),
		print >> fr2,'\r'
	for key in wordFreq:
		print >> fr3,key.encode('utf-8'),wordFreq[key]
	for i in label:
		print >> fr4,i

if __name__ == '__main__':
	trainDic,trainData,label = generateData(1)  #词典，每个样本的分词，类别，参数为类别所在的列
	wordFreq = wordFrequency(trainData) #生成词频
	saveData(trainDic,trainData,wordFreq,label) #save
