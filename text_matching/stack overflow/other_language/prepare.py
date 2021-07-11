from collections import Set

import pandas as pd
import jieba
import re
import io
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import word2vec

def preprocessing(data_df,year,fname):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:生成的预处理文件在preprocessing文件中
    做的工作：将句子中的脱敏数字***替换成一，句子中的错别字纠正，分词
    """


    # 加载停用词
    stopwords = load_stopwordslist("stop_words.txt")

    #当我使用普通的字典时，用法一般是dict={},添加元素的只需要dict[element] =value即，调用的时候也是如此，dict[element] = xxx,但前提是element字典里，如果不在字典里就会报错

    for index, row in data_df.iterrows():
        # 每1000个打印一下句子的词向量
        if index != 0 and index % 2000 == 0:
            print("{:,}  {}-sentence preprocessing.".format(index,fname))
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["s1_title", "s2_title",'s1_body','s2_body']:
            clean_str=re.sub(r"[^a-zA-Z ']","",str(row[col_name]))
            seg_str = seg_sentence(str(clean_str), stopwords)
            data_df.at[index, col_name] = seg_str

    data_df.to_csv('preprocessing/{}/mypair_{}.csv'.format(year,fname), sep='\t', header=None,index=None,encoding='utf-8')

def seg_sentence(sentence,stop_words):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    sentence_seged = jieba.cut(sentence.strip())#strip()方法，去除字符串开头或者结尾的空格
    out_str = ""
    for word in sentence_seged:
        if word not in stop_words:
            if word != " ":
                out_str += word
                out_str += " "
    out_str=out_str.strip()
    return out_str

def load_stopwordslist(filepath):
    """
    加载停用词
    :param filepath:停用词文件路径
    :return:
    """
    with io.open(filepath,"r",encoding="utf-8") as file:
        stop_words = [line.strip() for line in file]
        return stop_words

def divide(data_df,year,fname):


    X_df=data_df[['id','question1','question2','s1_title','s2_title','s1_body','s2_body','maintag']]
    Y_df=data_df[['duplicated']]
    X=X_df.values.tolist()
    Y=Y_df.values.tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

    X_train_df=pd.DataFrame(columns=['id','question1','question2','s1_title','s2_title','s1_body','s2_body','maintag'],index=None,data=X_train)
    X_test_df = pd.DataFrame(columns=['id', 'question1', 'question2', 's1_title', 's2_title', 's1_body', 's2_body','maintag'],index=None, data=X_test)
    Y_train_df=pd.DataFrame(columns=['duplicated'],data=Y_train)
    Y_test_df = pd.DataFrame(columns=['duplicated'], data=Y_test)


    X_train_df.insert(3, 'duplicated', Y_train_df)
    X_train_df.to_csv('divide/{}/mypair_{}_train.csv'.format(year,fname),encoding='utf-8',header=None,index=None, sep='\t')


    X_test_df.insert(3, 'duplicated', Y_test_df)



    X_test_df.to_csv('divide/{}/mypair_{}_test.csv'.format(year,fname),encoding='utf-8',index=None, sep='\t')

def stat(df):
    print(df.isnull().any())#检查是否有缺失值
    df.fillna("aa",inplace=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    s1_title_len=[]
    s2_title_len=[]
    s1_body_len=[]
    s2_body_len=[]

    for index,rows in df.iterrows():
        s1_title_len.append(len(df.loc[index,'s1_title'].split()))

        clean_body = ' '.join(df.loc[index,'s1_body'].split())
        s1_body_len.append(len(clean_body.split()))

    for index,rows in df.iterrows():
        s2_title_len.append(len(df.loc[index,'s2_title'].split()))
        clean_body = ' '.join(df.loc[index,'s2_body'].split())
        s2_body_len.append(len(clean_body.split()))


    sns.distplot(s1_body_len,ax=ax1,label='s1',color='blue')
    sns.distplot(s2_body_len,ax=ax2,label='s1',color='green')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax1.set_title('数据集S1长度分布')
    ax2.set_title('数据集S2长度分布')
    plt.show()
    sns.countplot(x='duplicated',data=df)
    plt.show()

    print("正负类别个数为{}".format(df['duplicated'].value_counts()))

file="mysql"
# data_df = pd.read_csv("data/2018/mypair_{}.csv".format(file), sep=',', header=0,names=["id","question1","question2","duplicated","s1_title","s2_title","s1_body","s2_body", "maintag"],encoding='utf-8-sig')
#
# preprocessing(data_df,2018,"{}_preprocessed".format(file))
# ##################################
# # print(len(data_df['question1']))
# # a=set(data_df['question1'])
# # b=set(data_df['question2'])
# #
# # print(len(a))
# # print(a&b)
# # print(len(a&b))
# #################################
#
# # #划分训练集测试集 8：2
data_df = pd.read_csv("./mypair_{}.csv".format(file), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
stat(data_df)
# divide(data_df,2018,file)
#
# #统计句子长度和正负例比例
# # data_df = pd.read_csv("divide/2018/mypair_ruby_train.csv", sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig')
# # stat(data_df)
#
# import pickle
# def save(nfile, object):
#     with open(nfile, 'wb') as file:
#         pickle.dump(object, file)


# data_df = pd.read_csv("divide/2018/mypair_{}_train.csv".format(file), sep='\t', header=None,names=["id","question1","question2","duplicated","s1_title","s2_title","s1_body","s2_body", "maintag"],encoding='utf-8-sig',lineterminator="\n")
#
# save("divide/2018/" + 'y_train_{}.pickle'.format(file), data_df['duplicated'].tolist())
#
#
# #######
# test_df= pd.read_csv("divide/2018/mypair_{}_test.csv".format(file), sep='\t', header=0,names=["id","question1","question2","duplicated","s1_title","s2_title","s1_body","s2_body", "maintag"],encoding='utf-8-sig',lineterminator="\n")
#
# save("data/2018/test/" + 'y_test_{}.pickle'.format(file), test_df['duplicated'].tolist())