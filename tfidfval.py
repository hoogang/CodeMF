import re
import numpy as np
from collections import Counter
import inflection #软件工程
from nltk import PorterStemmer
porter = PorterStemmer()
from nltk.corpus import stopwords


#缩略词处理
def abbrev(line):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
    # 's
    pat_s1 = re.compile("(?<=[a-zA-Z])\"s")
    # s
    pat_s2 = re.compile("(?<=s)\"s?")
    # not
    pat_not = re.compile("(?<=[a-zA-Z])n\"t")
    # would
    pat_would = re.compile("(?<=[a-zA-Z])\"d")
    # will
    pat_will = re.compile("(?<=[a-zA-Z])\"ll")
    # am
    pat_am = re.compile("(?<=[I|i])\"m")
    # are
    pat_are = re.compile("(?<=[a-zA-Z])\"re")
    # have
    pat_ve = re.compile("(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s1.sub("", line)
    line = pat_s2.sub("", line)
    line = pat_not.sub(" not", line)
    line = pat_would.sub(" would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)
    new_line = line.replace('\'', ' ')

    return new_line

#程序类型分词
def tokenize(text):
    abbrev_text = abbrev(text)
    proed_text = inflection.underscore(abbrev_text)
    word_stem = [porter.stem(w) for w in re.findall(r"[\w'-]+|[^\s\w]", proed_text)]
    words = [ i for i in word_stem if i not in stopwords.words('english')]
    return  words

def tf(word, count):
    return count[word] / sum(count.values())

def n_containing(word, count_list):
    return np.sum([1 for count in count_list if word in count])

def idf(word, count_list):
    return np.log(len(count_list) / (1 + n_containing(word, count_list)))+1

def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def count(corpus_list):
    words_list=[tokenize(line) for line in corpus_list]
    scores=[]
    #语料转成语句
    count_list  = [Counter(i) for i in words_list]
    for i, count in enumerate(count_list):
        #计算逆文档频率
        sums=np.sqrt(np.sum([np.square(tfidf(word, count, count_list)) for word in count]))
        #计算
        words_tfidf  = {word: tfidf(word, count, count_list)/sums for word in count}
        #字典value
        tfidf_score=np.sum(list(words_tfidf.values()))
        scores.append(tfidf_score)

    scores=np.asarray(scores)
    #归一化操作
    norm_scores=(scores-scores.min())/(scores.max()-scores.min())

    return norm_scores









