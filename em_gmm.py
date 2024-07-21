import math
import numpy
from numpy import random

pi = math.pi
exp = math.exp
sqrt = math.sqrt
log = math.log


def gaussD(x, mu, sigma):
	if sigma < 5:
		sigma = 3 + 2* random.random()
	return exp(-1 * (x - mu) * (x - mu) / 2 / sigma) / sqrt(2 * pi * sigma)


def eStep(dataList, aList, muList, sigList, enlar=1e1):
	E_MAX = []
	rows = len(dataList[0])
	cols = len(dataList)
	gaussMatrix = []
	rowSumList = []
	Ln = 1
	for j in range(cols):
		gaussMatrix_j = []
		for i in range(rows):
			tmp = aList[j] * gaussD(dataList[j][i], muList[j], sigList[j]) * enlar
			gaussMatrix_j.append(tmp)
		gaussMatrix.append(gaussMatrix_j)

	for i in range(rows):
		rowSum = 0
		for j in range(cols):
			rowSum += gaussMatrix[j][i]
		Ln *= rowSum
		if rowSum == 0:
			print(i, j)
		rowSumList.append(rowSum)

	for j in range(cols):
		E_MAX_j = []
		for i in range(rows):
			tmpSum = rowSumList[i]

			if tmpSum == 0:
				wij = 1 / cols
			else:
				wij = gaussMatrix[j][i] / tmpSum
			E_MAX_j.append(wij)
		E_MAX.append(E_MAX_j)
	return Ln, E_MAX


def mStep(E_MAX, dataList, aList, muList, sigList):
	rows = len(dataList[0])
	cols = len(dataList)

	nAList = []
	nMuList = []
	nSigList = []

	for j in range(cols):
		sum_alp = 0
		sum_mu = 0
		sum_sig = 0
		for i in range(rows):
			sum_alp += E_MAX[j][i]
			sum_mu += E_MAX[j][i] * dataList[j][i]
			sum_sig += E_MAX[j][i] * (dataList[j][i] - muList[j]) * (dataList[j][i] - muList[j])
		nA = sum_alp / rows
		nAList.append(nA)
		if (aList[j] == 0):
			nMu = 0
			nSig = 1
		else:
			nMu = sum_mu / (rows * aList[j])
			nSig = sum_sig / (rows * aList[j])
		nMuList.append(nMu)

		nSigList.append(nSig)

	return nAList, nMuList, nSigList


def getMuAndSigma(list):
	mu = sum(list) / len(list)
	sigma = 0
	for i in list:
		sigma += (i - mu) * (i - mu)
	sigma = sigma / len(list)
	return mu, sigma


def getMSForRequired(datalist):
	muList = []
	sigList = []
	le = len(datalist)
	for i in range(le):
		mu, sigma = getMuAndSigma(datalist[i])
		muList.append(mu)
		sigList.append(sigma)
	return muList, sigList


def estimate(attribution_list):
	le = len(attribution_list[0])  # 属性的个数
	max_iter = 1000
	threshold = 1e-10
	data = numpy.array(attribution_list).T

	le = len(data)
	alp= []
	for i in range(le):  # 初始化
		alp.append(1 / le)
	muList, sigList = getMSForRequired(data)
	count = 0
	oLn = 0
	muMatrix = [muList]
	sigMatrix = [sigList]
	aMatrix = [alp]
	while count < max_iter:
		count += 1
		Ln, E_Max = eStep(data, alp, muList, sigList)
		newA, newMuList, newSigList = mStep(E_Max, data, alp, muList, sigList)

		aCha = 0
		muCha = 0
		sigCha = 0
		for i in range(le):
			aCha += (newA[i] - alp[i]) * (newA[i] - alp[i])
			muCha += (newMuList[i] - muList[i]) * (newMuList[i] - muList[i])
			sigCha += (newSigList[i] - muList[i]) * (newSigList[i] - muList[i])
		if aCha / le < threshold or muCha / le < threshold or sigCha / le < threshold:
			break
		oLn = Ln
		alp = newA
		muList = newMuList
		sigList = newSigList
		muMatrix.append(muList)
		sigMatrix.append(sigList)
	return alp


