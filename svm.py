import re
import nltk
from nltk import PorterStemmer
porter = PorterStemmer()
from collections import OrderedDict
import numpy as np
from sklearn import svm



class Classfier:

    @staticmethod
    def tokenize(text):
        text = text.lower()
        # split punctuations but dont split single quotes for words like don't
        word = [porter.stem(w) for w in re.findall(r"[\w'-]+|[^\s\w]", text)]
        biword =  [b for b in nltk.bigrams(word)]
        triword = [t for t in nltk.trigrams(word)]
        # word = [w for w in word if w not in stopwords.words('english')]
        return  word # triword


    def train(self, posTrainCorpus, negTrainCorpus):
        tokens = []

        fp = open(posTrainCorpus, 'r',encoding='UTF-8')
        for line in fp:
          tokens += Classfier.tokenize(line)
        fp.close()

        fn = open(negTrainCorpus, 'r',encoding='UTF-8')
        for line in fn:
          tokens += Classfier.tokenize(line)
        fn.close()

        #Create Frequency Distribution from both Positive and Negative Corpora
        trainFreq = nltk.FreqDist(tokens)

        #No of Features
        self.noFeat = len(trainFreq)

        #Get Keys to maintain Order
        self.trainKeys = trainFreq.keys()

        #Create OrderedDict for features: Use this as sample for all files
        ordFeat = OrderedDict()
        for key in trainFreq.keys():
            ordFeat.update( {key: trainFreq.freq(key)} )

        posFeatList = self.featureList(posTrainCorpus)
        negFeatList = self.featureList(negTrainCorpus)
        featList = posFeatList + negFeatList

        noPos = len(posFeatList)
        noNeg = len(negFeatList)

        labels = []

        for j in range(noPos): #0 正
            labels.append(0)
        for k in range(noNeg):
            labels.append(1)   #1 负

        #Create numpy Array for word frequencies : Feature Vector
        trainFreqArr = np.array(featList)
        trainLabels = np.array(labels)


        #Fit SVM
        self.docClassifier = svm.LinearSVC()
        self.docClassifier.fit(trainFreqArr, trainLabels)


    def getFeat(self, line):
        listItem = [0]*self.noFeat
        fileFreqDist = nltk.FreqDist(Classfier.tokenize(line))

        i = 0
        for key in self.trainKeys:
            if key in fileFreqDist.keys():
                listItem[i] = fileFreqDist.get(key)
            i = i + 1
        return listItem

    def featureList(self, corpus):
        featList = []
        f = open(corpus, 'r',encoding='UTF-8')
        for line in f:
            featList.append(self.getFeat(line))
        f.close()

        return featList


    def test(self, posTestCorpus, negTestCorpus):
        posTestFeatList = self.featureList(posTestCorpus)
        negTestFeatList = self.featureList(negTestCorpus)

        posTestarr = np.array(posTestFeatList)
        negTestarr = np.array(negTestFeatList)


        # prediction result stored in array which is the converted to list and added to opt list
        print("Good rate =" + str((116-sum(np.array(self.docClassifier.predict(posTestarr)).tolist())) / 116.0))
        print("Bad rate = " + str(sum(np.array(self.docClassifier.predict(negTestarr)).tolist()) / 116.0))
        print("Accuracy  = " + str((116-(sum(np.array(self.docClassifier.predict(posTestarr)).tolist())) + sum(np.array(self.docClassifier.predict(negTestarr)).tolist() )) /232.0))

    def filter(self, sent):
        testFeatList = []
        testFeatList.append(self.getFeat(sent))
        testarr = np.array(testFeatList)
        opt = np.array(self.docClassifier.predict(testarr)).tolist()
        return opt[0]


posTrainCorpus="balanced/pos_train.txt"
negTrainCorpus="balanced/neg_train.txt"
posTestCorpus= "balanced/pos_test.txt"
negTestCorpus= "balanced/neg_test.txt"

if __name__ == '__main__':
    s = Classfier()
    s.train(posTrainCorpus, negTrainCorpus)
    s.test(posTestCorpus, negTestCorpus)
    print(s.filter("Return the number of characters in two strings that don't exactly match"))  #0  正
    print(s.filter("Return True if object is defined.")) #1 负
