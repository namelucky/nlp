import math

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
from keras.preprocessing import sequence


def read(language):
    test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format(language), sep='\t',
                               names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                  "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n", header=0)
    train_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format(language), sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                  "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    print(train_df.info())
    print(train_df.head())
    print(test_df.info())
    print(test_df.head())
    train_df.fillna(language,inplace=True)
    test_df.fillna(language,inplace=True)
    texts_s1_test = test_df['s1_title'].tolist()
    texts_s2_test = test_df['s2_title'].tolist()

    texts_s1_train = train_df['s1_title'].tolist()
    texts_s2_train = train_df['s2_title'].tolist()
    texts=[]

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)
    print(np.array(texts).shape)
    return texts_s1_train,texts_s1_test,texts


def seg(content, stopwords):
    '''
    分词并去除停用词
    '''
    segs = jieba.cut(content, cut_all=True)
    segs = [w.encode('utf8') for w in list(segs)]# 特别注意此处转换

    seg_set = set(set(segs) - set(stopwords))
    return seg_set

def docs(w, D):
    c = 0
    for d in D:
        if w in d:
            c = c + 1;
    return c

def save(idf_dict, path):
    # with open(path, "a+") as f:
    #     f.truncate()
    #     # write_list = []
    #     for key in idf_dict.keys():
    #         # write_list.append(str(key)+" "+str(idf_dict[key]))
    #         f.write(str(key) + " " + str(idf_dict[key]) + "\n")
    with open(path,'wb') as f:
        pickle.dump(idf_dict,f)


def compute_idf(texts, stopwords):
    # 所有分词后文档
    D = []
    #所有词的set
    W = set()
    for i in range(len(texts)):
        #新闻原始数据
        prevue = texts[i]
        d = seg(prevue, stopwords)
        D.append(d)
        W = W | d
    #计算idf
    idf_dict = {}
    n = len(W)
    #idf = log(n / docs(w, D))
    for w in list(W):
        idf = math.log(n*1.0 / docs(w, D))
        idf_dict[str(w,encoding='utf-8')] = idf
    return idf_dict


def pad(l,maxlen=20):
    if(len(l)>maxlen):
        return l[0:maxlen]
    else:
        while(len(l)<maxlen):
            l.insert(0,0)
        return l

def title_sequence(title,idf_dict,path):
    all=[]
    for i in range(len(title)):
        l=title[i]
        t=[]
        for word in l.split(" "):
            if word in idf_dict:
                t.append(idf_dict[word])
        t=pad(t)
        all.append(t)
    print(np.array(all).shape)
    with open(path,'wb') as f:
        pickle.dump(all,f)
    # print(np.array(all).shape)


language='objective-c'
stopwords={}
texts_s1_train,texts_s1_test,texts=read(language)
idf_dict = compute_idf(texts, stopwords)
# # #存储

path = "./{}/idf.pickle".format(language)

l=language
if language == 'objective-c':
    path = "./c/idf.pickle"
    l='c'

save(idf_dict, path)


with open(path,'rb') as f:
    idf_dict=pickle.load(f)
# print(len(texts))
title_sequence(texts_s1_train,idf_dict,'./{}/{}_train_idf.pickle'.format(l,l))
title_sequence(texts_s1_test,idf_dict,'./{}/{}_test_idf.pickle'.format(l,l))