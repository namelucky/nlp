#基于词向量计算相似度的有使用词向量求平均计算相似度和词向量tf-idf加权求平均相似度等几种方法
import numpy as np
import os
import gensim
from scipy.linalg import norm
import pandas as pd
import pickle
import string
from gensim.models import KeyedVectors
from smart_open import smart_open

from sklearn.metrics import f1_score,accuracy_score


train_original = '../data/2018/train/train.csv'#训练集的文件名
test_original= '../data/2018/train/test.csv'

w2v_path = '../w2c.bigram'
w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)


def sentence_vector(s):
    '''
    输入句子，获得该句子对应的句向量
    :param s: 句子
    :return:
    '''
    v = np.zeros(300)
    length = len(s)
    for word in s:
        try:
            v += w2v_model[word]
        except:
            length -= 1
            continue
    v /= length
    return v



def similarity(s1, s2):
    s1=s1.strip().split(" ")
    s2 = s2.strip().split(" ")

    v1 = sentence_vector(s1)
    v2 = sentence_vector(s2)

    sim = np.dot(v1, v2) / (norm(v1) * norm(v2))  # 计算得到的相似度

    return v1,v2,sim


def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


if __name__ == '__main__':

    train_w2v=[]
    test_w2v=[]

    train_body1=[]#存放body 的word2vec向量
    train_body2=[]

    test_body1=[]
    test_body2=[]


    train_df = pd.read_csv(train_original,
                           names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                  "s2_body", "maintag"], header=None, encoding='utf-8-sig', sep='\t',
                           lineterminator="\n")
    train_df.fillna('33', inplace=True)

    test_df = pd.read_csv(test_original,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], header=None, encoding='utf-8-sig', sep='\t',
                          lineterminator="\n")
    test_df.fillna('33', inplace=True)

    for index, row in train_df.iterrows():
        s1_vec,s2_vec,cos = similarity(train_df.at[index, 's1_body'], train_df.at[index, 's2_body'])
        train_w2v.append(cos)
        train_body1.append(s1_vec)
        train_body2.append(s2_vec)

    save('train_w2v_body1.pickle', train_body1)
    save('train_w2v_body2.pickle', train_body2)

    train_w2v = np.array(train_w2v) > 0.5
    train_df['w2c_pre'] = train_w2v.astype(int)
    print('w2c训练集f1', f1_score(train_df['duplicated'], train_df['w2c_pre']))
    print('w2c训练集ACC', accuracy_score(train_df['duplicated'], train_df['w2c_pre']))

    for index, row in test_df.iterrows():
        s1_vec,s2_vec,cos = similarity(test_df.at[index, 's1_body'], test_df.at[index, 's2_body'])
        test_w2v.append(cos)
        test_body1.append(s1_vec)
        test_body2.append(s2_vec)

    save('test_w2v_body1.pickle', test_body1)
    save('test_w2v_body2.pickle', test_body2)

    test_w2v = np.array(test_w2v) > 0.5
    test_df['w2c_pre'] = test_w2v.astype(int)
    print('w2c训练集f1', f1_score(test_df['duplicated'], test_df['w2c_pre']))
    print('w2c训练集ACC', accuracy_score(test_df['duplicated'], test_df['w2c_pre']))
