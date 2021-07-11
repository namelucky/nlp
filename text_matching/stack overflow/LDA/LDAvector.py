import math

from gensim import corpora,models
import jieba
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import pickle


train_original = '../data/2018/train/train.csv'#训练集的文件名
test_original= '../data/2018/train/test.csv'

def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)

def train():
    # documents = ['Python 是 目前 最 流行 的 数据分析 和 机器 学习 编程语言',
    #  'Python 语言 编程 将 很快 成为 各个 高校 的 必修课',
    #  'Python 是 科研 工作者 开展 科学研究 的 高效 工具']
    texts = []
    train_df = pd.read_csv(train_original,
                                    names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                                           "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig',
                                    sep='\t', lineterminator="\n")
    train_df.fillna('33', inplace=True)  # title经过去除数字的预处理 产生了3条缺失值，我随机填充成数字333

    test_df = pd.read_csv(test_original,
                                   names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                                          "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig', sep='\t',
                                   lineterminator="\n")
    test_df.fillna('33', inplace=True)

    train_body = train_df['s1_body'].tolist()
    train_body2 = train_df['s2_body'].tolist()
    test_body = test_df['s1_body'].tolist()
    test_body2 = test_df['s2_body'].tolist()

    texts.extend(train_body)
    texts.extend(train_body2)
    texts.extend(test_body)
    texts.extend(test_body2)


    raw_corpus = texts


    documents = []

    for sentence in raw_corpus:
        # sentence = ''.join(re.findall(r'[\u4e00-\u9fa5]+', sentence))  # 仅保留中文
        stop_words = [' ']  # 取出停用词
        documents.append([item for item in jieba.cut(sentence) if item not in stop_words])  # 去掉停止词


    texts = documents
        # [[word for word in document.lower().split() ] #删除常用单词（使用停止词列表）
        #      for document in documents]
    print(texts)
    dictionary = corpora.Dictionary(texts) #创建一个映射字典
    dictionary.save('./deerwester.dict')
    print(dictionary)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('./deerwester.mm', corpus)

    lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=30,minimum_probability=0.0)
    lda.save('./model.lda')


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(vec_a, vec_b):
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)

def similarity(s1, s2):
    s1_vec = dictionary.doc2bow(s1.lower().split())  # 使用我们刚构造的字典来进行编码
    s1_vec = lda[s1_vec]
    s1_vec = [i[1] for i in s1_vec]

    s2_vec = dictionary.doc2bow(s2.lower().split())  # 使用我们刚构造的字典来进行编码
    s2_vec = lda[s2_vec]
    s2_vec = [i[1] for i in s2_vec]

    cos = cosine_similarity(s1_vec, s2_vec)  # 余弦相似度
    return s1_vec,s2_vec,cos

def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)




train()
# print('finish train')
lda=models.LdaModel.load('./model.lda')
dictionary=corpora.Dictionary.load('./deerwester.dict')
#
#
#
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

train_lda=[]#存放lda向量的余弦相似度 训练集
test_lda=[]#存放lda向量的余弦相似度 测试集

train_body1=[]#存放lda向量
train_body2=[]

test_body1=[]
test_body2=[]

for index, row in train_df.iterrows():
    s1_vec,s2_vec,cos=similarity(train_df.at[index, 's1_body'],train_df.at[index, 's2_body'])
    train_lda.append(cos)
    train_body1.append(s1_vec)
    train_body2.append(s2_vec)


save('train_lda.pickle',train_lda)
save('train_lda_body1.pickle',train_body1)
save('train_lda_body2.pickle',train_body2)


# train_lda=load('./train_lda.pickle')


train_lda=np.array(train_lda)>0.5
train_df['lda_pre']=train_lda.astype(int)
print('LDA训练集f1',f1_score(train_df['duplicated'],train_df['lda_pre']))
print('LDA训练集ACC',accuracy_score(train_df['duplicated'],train_df['lda_pre']))

#
for index, row in test_df.iterrows():
    s1_vec,s2_vec,cos=similarity(test_df.at[index, 's1_body'], test_df.at[index, 's2_body'])
    test_lda.append(cos)
    test_body1.append(s1_vec)
    test_body2.append(s2_vec)

save('test_lda.pickle',test_lda)
save('test_lda_body1.pickle',test_body1)
save('test_lda_body2.pickle',test_body2)

test_lda=np.array(test_lda)>0.5
test_df['lda_pre']=test_lda.astype(int)
print('LDA训练集f1',f1_score(test_df['duplicated'],test_df['lda_pre']))
print('LDA训练集ACC',accuracy_score(test_df['duplicated'],test_df['lda_pre']))
