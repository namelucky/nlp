from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import math
import pickle
from sklearn.metrics import f1_score,accuracy_score
import numpy as np

# data_df = pd.read_csv("../data/preprocess_data/mypair_{}_preprocessed.csv".format('ruby'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
train_original = '../data/2018/train/train.csv'#训练集的文件名
test_original= '../data/2018/train/test.csv'
train_df = pd.read_csv(train_original,
                       names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                              "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig',
                       sep='\t', lineterminator="\n")
test_df = pd.read_csv(test_original,
                      names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                             "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig', sep='\t',
                      lineterminator="\n")

train_df['s2_body'].fillna('aa', inplace=True)
train_df['s1_body'].fillna('aa', inplace=True)

test_df['s2_body'].fillna('aa', inplace=True)
test_df['s1_body'].fillna('aa', inplace=True)

texts=[]
for index, rows in train_df.iterrows():
    texts.append(train_df.loc[index,'s1_body'].split())
    texts.append(train_df.loc[index, 's2_body'].split())

for index, rows in test_df.iterrows():
    texts.append(test_df.loc[index,'s1_body'].split())
    texts.append(test_df.loc[index, 's2_body'].split())

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, vector_size=30, window=2, min_count=1, workers=4)

from gensim.test.utils import get_tmpfile

fname = get_tmpfile("my_doc2vec_model")
model.save(fname)
model = Doc2Vec.load(fname)

def norm(vector):
    return math.sqrt(sum(x * x for x in vector))

def similarity(s1,s2):

    vector1 = model.infer_vector(s1.split())
    vector2 = model.infer_vector(s2.split())
    norm_a = norm(vector1)
    norm_b = norm(vector2)
    dot = sum(a * b for a, b in zip(vector1, vector2))
    return vector1,vector2,dot / (norm_a * norm_b)

def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


train_d2v=[]#存放doc2vec向量的余弦相似度 训练集
test_d2v=[]#存放doc2vec向量的余弦相似度 测试集

train_body1=[]#存放doc2vec向量
train_body2=[]

test_body1=[]
test_body2=[]


for index, rows in train_df.iterrows():
    s1_vec,s2_vec,cos=similarity(train_df.loc[index,'s1_body'],train_df.loc[index,'s2_body'])
    train_d2v.append(cos)
    train_body1.append(s1_vec)
    train_body2.append(s2_vec)

save('train_doc2vec.pickle',train_d2v)
save('train_doc2vec_body100.pickle',train_body1)
save('train_doc2vec_body200.pickle',train_body2)


train_d2v=np.array(train_d2v)>0.5
train_df['lda_pre']=train_d2v.astype(int)
print('Doc2vec训练集f1',f1_score(train_df['duplicated'],train_df['lda_pre']))
print('Doc2vec训练集ACC',accuracy_score(train_df['duplicated'],train_df['lda_pre']))

#
for index, row in test_df.iterrows():
    s1_vec,s2_vec,cos=similarity(test_df.at[index, 's1_body'], test_df.at[index, 's2_body'])
    test_d2v.append(cos)
    test_body1.append(s1_vec)
    test_body2.append(s2_vec)

save('test_doc2vec.pickle',test_d2v)
save('test_doc2vec_body100.pickle',test_body1)
save('test_doc2vec_body200.pickle',test_body2)

test_lda=np.array(test_d2v)>0.5
test_df['lda_pre']=test_lda.astype(int)
print('doc2vec测试集f1',f1_score(test_df['duplicated'],test_df['lda_pre']))
print('doc2vec测试集ACC',accuracy_score(test_df['duplicated'],test_df['lda_pre']))
