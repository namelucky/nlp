import pandas as pd
from gensim.models import word2vec
import pickle
import numpy as np

def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)
def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)

def mergetrain():
    '''
    合并训练集，训练神经网络模型
    '''
    java_df = pd.read_csv("divide/2018/mypair_java_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    print("java行数 {}".format(java_df.shape[0]))
    java_y_se=load("divide/2018/y_train_java.pickle")
    print("java label行数 {}".format(len(java_y_se)))

    python_df = pd.read_csv("divide/2018/mypair_python_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    python_y_se=load("divide/2018/y_train_python.pickle")
    print("python行数 {}".format(python_df.shape[0]))
    print("python label行数 {}".format(len(python_y_se)))

    c_df = pd.read_csv("divide/2018/mypair_c++_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    c_y_se=load("divide/2018/y_train_c++.pickle")
    print("c++行数 {}".format(c_df.shape[0]))
    print("c++ label行数 {}".format(len(c_y_se)))

    ruby_df = pd.read_csv("divide/2018/mypair_ruby_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    ruby_y_se=load("divide/2018/y_train_ruby.pickle")
    print("ruby行数 {}".format(ruby_df.shape[0]))
    print("ruby label行数 {}".format(len(ruby_y_se)))
    html_df=pd.read_csv("divide/2018/mypair_html_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    html_y_se = load("divide/2018/y_train_html.pickle")

    objc_df=pd.read_csv("divide/2018/mypair_objective-c_train.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    objc_y_se = load("divide/2018/y_train_objective-c.pickle")

    java_df=java_df.append([python_df,c_df,ruby_df,html_df,objc_df],ignore_index=True)


    java_y_se.extend(python_y_se)
    java_y_se.extend(c_y_se)
    java_y_se.extend(ruby_y_se)
    java_y_se.extend(html_y_se)
    java_y_se.extend(objc_y_se)

    java_df.to_csv("data/2018/train/train.csv",header=None,index=None,sep='\t')
    save('data/2018/train/y_train.pickle', java_y_se)
    print(len(java_y_se))
    print(java_df.info())

def saveLabel(file,flag="test"):
    '''
    将label存储为pickle文件
    flag表示train or test
    file表示 java or c++这些
    '''
    data_df = pd.read_csv("divide/2018/mypair_{}_{}.csv".format(file,flag), sep='\t', header=0,names=["id","question1","question2","duplicated","s1_title","s2_title","s1_body","s2_body", "maintag"],encoding='utf-8-sig',lineterminator="\n")

    save("divide/2018/" + 'y_{}_{}.pickle'.format(flag,file), data_df['duplicated'].tolist())
def saveTrainLabel(file,flag="train"):
    '''
    将label存储为pickle文件
    flag表示train or test
    file表示 java or c++这些
    '''
    data_df = pd.read_csv("divide/2018/mypair_{}_{}.csv".format(file,flag), sep='\t', names=["id","question1","question2","duplicated","s1_title","s2_title","s1_body","s2_body", "maintag"],encoding='utf-8-sig',lineterminator="\n")
    save("divide/2018/" + 'y_{}_{}.pickle'.format(flag,file), data_df['duplicated'].tolist())


def mergetest():
    '''
    合并测试集
    :return:
    '''
    java_test_df = pd.read_csv("divide/2018/mypair_java_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)
    print("java行数 {}".format(java_test_df.shape[0]))



    python_test_df = pd.read_csv("divide/2018/mypair_python_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)

    print("python行数 {}".format(python_test_df.shape[0]))


    c_test_df = pd.read_csv("divide/2018/mypair_c++_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)

    print("c++行数 {}".format(c_test_df.shape[0]))


    ruby_test_df = pd.read_csv("divide/2018/mypair_ruby_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)

    print("ruby行数 {}".format(ruby_test_df.shape[0]))

    html_test_df=pd.read_csv("divide/2018/mypair_html_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)
    objc_test_df=pd.read_csv("divide/2018/mypair_objective-c_test.csv", sep='\t',
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n",header=0)

    java_test_df=java_test_df.append([python_test_df,c_test_df,ruby_test_df,html_test_df,objc_test_df],ignore_index=True)



    java_test_df.to_csv("data/2018/train/test.csv",header=None,index=None,sep='\t')

    print(java_test_df.info())

    java_y_se=load("divide/2018/y_test_java.pickle")
    python_y_se=load("divide/2018/y_test_python.pickle")
    c_y_se=load("divide/2018/y_test_c++.pickle")
    ruby_y_se=load("divide/2018/y_test_ruby.pickle")
    html_y_se=load("divide/2018/y_test_html.pickle")
    objc_y_se=load("divide/2018/y_test_objective-c.pickle")
    java_y_se.extend(python_y_se)
    java_y_se.extend(c_y_se)
    java_y_se.extend(ruby_y_se)
    java_y_se.extend(html_y_se)
    java_y_se.extend(objc_y_se)
    print(np.array(java_y_se).shape)
    save('data/2018/train/y_test.pickle', java_y_se)



def merge2():
    '''
    合并数据集训练词向量
    :return:
    '''
    java_df = pd.read_csv("data/2018/mypair_java.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    #print("java行数 {}".format(java_df.shape[0]))
    print(java_df)
    python_df = pd.read_csv("data/2018/mypair_python.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    c_df = pd.read_csv("data/2018/mypair_c++.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    ruby_df = pd.read_csv("data/2018/mypair_ruby.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    html_df=pd.read_csv("data/2018/mypair_html.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    objc_df=pd.read_csv("data/2018/mypair_objective-c.csv", sep=',',header=0,
                          names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body",
                                 "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
    java_df=java_df.append([python_df, c_df, ruby_df,html_df,objc_df], ignore_index=True)


    java_df.to_csv("data/2018/all/all.csv", header=None, index=None, sep='\t')

def train_word_w2v(data_df,f_name,binary = False):
    """
    训练词向量
    :param file_name:需要进行训练词向量的文本文件名字，位于preprocessing目录中
    :param f_name:生成的词向量的文件名
    :param binary:将词向量表是否存储为二进制文件
    :return:
    """
    # 加载所有的词汇表训练集和测试集
    texts = []
    texts_s1_train = [line.strip().split(" ") for line in data_df['s1_title'].tolist()]
    texts_s2_train = [line.strip().split(" ") for line in data_df['s2_title'].tolist()]
    texts_s3_train = [line.strip().split(" ") for line in data_df['s1_body'].tolist()]
    texts_s4_train = [line.strip().split(" ") for line in data_df['s2_body'].tolist()]
    texts.extend(texts_s1_train)#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    texts.extend(texts_s2_train)
    texts.extend(texts_s3_train)
    texts.extend(texts_s4_train)
    model = word2vec.Word2Vec(sentences=texts,size=300,window=2,min_count=3,workers=2)
    #sentences：可以是一个list，是训练预料，min_count是小于该数的单词会被踢出，默认值为5，window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少， workers：用于控制训练的并行数
    model.wv.save_word2vec_format(fname=f_name,binary=binary,fvocab=None)

    #

def savePickle():
    '''
    将标签全部存储为pickle文件
    :return:
    '''
    saveTrainLabel("java","train")##存储label的函数
    saveTrainLabel("c++","train")
    saveTrainLabel("html","train")
    saveTrainLabel("python","train")
    saveTrainLabel("ruby","train")
    saveTrainLabel("objective-c","train")

    saveLabel("java","test")
    saveLabel("c++","test")
    saveLabel("html","test")
    saveLabel("python","test")
    saveLabel("ruby","test")
    saveLabel("objective-c","test")

savePickle()#dataframe中的duplicated列单独取出来 ，存成pickle文件
# print('merge train')
mergetrain()#合并训练集的语料和 label

mergetest()#//合并测试集的语料和label

# # print('merge词向量')

merge2()#所有的数据合并 输出all.csv用来训练词向量
data_df = pd.read_csv("data/2018/all/all.csv", sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig', lineterminator="\n")
# # print(data_df)
train_word_w2v(data_df,"w2c.bigram")




