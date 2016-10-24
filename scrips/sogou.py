# -*- coding:utf-8 -*-

import numpy
import jieba
import re

def generateData(labelDim):  
	fr = open('../datas/user_tag_query.2W.TEST')
	label = []
	m = 1
	trainSet = []
	trainData = []
	for line in fr.readlines():  #遍历每行	
		#print m
		line = line.strip().split()
		n = len(line)
		#label.append(int(line[labelDim]))
		tmpData = []
		for i in range(4,n):  #遍历该用户的所有词条
			#tmp = line[i].decode('GBK')	
			tmp = line[i]
			tmp = jieba.cut_for_search(tmp)
			for j in tmp:  #该词条的分词列表
				if len(j) > 1 and (re.match(r'^[+-]?\d+(.\d*)*$',j) is None):  #过滤掉单个字和纯数字（包括带小数点）
					trainSet.append(j)
					tmpData.append(j)
		trainData.append(tmpData)
		m += 1
	trainSet = set(trainSet)		
	return trainSet,trainData
	
def loadData():  #导入词典 分词矩阵，label
	fr1 = open('testDic.txt','r')
	fr2 = open('testData.txt','r')
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
	return trainDic,trainData
	
def wordFrequency(trainDic,trainData): #计算词频
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

def saveData(trainDic,trainData,wordFreq):  #将生成的词典 分词样本 词频 label离线保存
	fr1 = open('testDic.txt','w')
	fr2 = open('testData.txt','w')
	fr3 = open('testFreq.txt','w')
	for key in trainDic:
		print >> fr1,key.encode('utf-8')
	for i in range(len(trainData)):
		for j in range(len(trainData[i])):
			print >> fr2,trainData[i][j].encode('utf-8'),
		print >> fr2,'\r'
	for key in wordFreq:
		print >> fr3,key.encode('utf-8'),wordFreq[key]

if __name__ == '__main__':
	trainDic,trainData = generateData(1)  #词典，每个样本的分词，类别，参数为类别所在的列
	wordFreq = wordFrequency(trainDic,trainData) #生成词频
	saveData(trainDic,trainData,wordFreq) #save
