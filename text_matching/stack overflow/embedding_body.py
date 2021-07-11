import pandas as pd
import json
import re
from collections import defaultdict
from gensim.models import word2vec
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import jieba
import numpy as np
import os
import pickle

train_original = 'data/2018/train/train.csv'  # 训练集的文件名

embedding_catalog = 'data/2018/embedding_body'  # 做完embedding操作，文件存放的文件夹
embedding_size = 300  # 词向量的维度
max_sentence_length = 20  # 词级别的向量padding的长度
max_vovab_size = 100000  # tokenizer的numword
max_body_length=200


def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


def embedding_word(f_name, train_df, test_df, train_filename, test_filename, type=2):
    """
    :param f_name 词嵌入矩阵的名字
    :param type: 词向量的选择：1，知乎语料词向量，2，训练集训练的词向量 3 知乎+训练集
    :return:
    """
    if type == 1:
        w2v_path = 'sgns.zhihu.bigram'
    elif type == 2:
        w2v_path = 'w2c.bigram'
    elif type == 3:
        w2v_path = 'sgns.zhihu.bigram'
        w2v_path2 = 'w2c.bigram'
        w2v_model2 = KeyedVectors.load_word2vec_format(w2v_path2, binary=False)

    # 加载预训练的词向量w2v
    print('load w2v_model...')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    print('finish w2v_model...')

    tokenizer = Tokenizer(
        num_words=max_vovab_size,
        split=' ',
        lower=False,
        char_level=False,
        filters=''
    )

    texts = []
    texts_s1_test = test_df['s1_title'].tolist()
    texts_s2_test = test_df['s2_title'].tolist()

    texts_body1_test = test_df['s1_body'].tolist()
    texts_body2_test = test_df['s2_body'].tolist()

    texts_s1_train = train_df['s1_title'].tolist()
    texts_s2_train = train_df['s2_title'].tolist()

    texts_body1_train = train_df['s1_body'].tolist()
    texts_body2_train = train_df['s2_body'].tolist()

    texts.extend(texts_s1_test)
    texts.extend(texts_s2_test)
    texts.extend(texts_s1_train)
    texts.extend(texts_s2_train)
    texts.extend(texts_body1_test)
    texts.extend(texts_body2_test)
    texts.extend(texts_body1_train)
    texts.extend(texts_body2_train)

    tokenizer.fit_on_texts(texts)

    num_words_dict = tokenizer.word_index

    # 训练集的词汇表的词向量矩阵,行数为最大值+1,列数为词向量的维度，形式为：index->vec，主要是为了构建单词在字典中的序号index->单词的词向量
    embedding_matrix = 1 * np.random.randn(len(num_words_dict) + 1, embedding_size)
    print('keras的embedding矩阵维度为[{},{}]'.format(len(num_words_dict) + 1, embedding_size))
    embedding_matrix[0] = np.random.randn(embedding_size)

    count = 0
    nocount = 0
    for word, index in num_words_dict.items():  # index是从1开始的
        if word in w2v_model.vocab:
            embedding_matrix[index] = w2v_model.word_vec(word)
            count = count + 1
        else:
            if type == 3:
                if word in w2v_model2.vocab:
                    embedding_matrix[index] = w2v_model2.word_vec(word)
                    count = count + 1
                else:
                    print('{}没有进行向量化'.format(word))
                    nocount = nocount + 1;
    print('total {}, word in model have {}'.format(len(num_words_dict), count))
    print('有{}个词没有进行embedding向量化'.format(nocount))
    save(f_name, embedding_matrix)  # 词嵌入矩阵
    print('finish')
    print('将对应文本转为数字，并进行padding操作')
    # 将文本向量化，将单词转为对应的字典的数字  ['ha ha gua angry'] {{'ha': 1, 'gua': 2, 'angry': 3,.....},文本就会变成[1, 1, 2, 3],
    s1_train_sequence = tokenizer.texts_to_sequences(texts_s1_train)
    s2_train_sequence = tokenizer.texts_to_sequences(texts_s2_train)
    s1_test_sequence = tokenizer.texts_to_sequences(texts_s1_test)
    s2_test_sequence = tokenizer.texts_to_sequences(texts_s2_test)

    body1_train_sequence = tokenizer.texts_to_sequences(texts_body1_train)
    body2_train_sequence = tokenizer.texts_to_sequences(texts_body2_train)
    body1_test_sequence = tokenizer.texts_to_sequences(texts_body1_test)
    body2_test_sequence = tokenizer.texts_to_sequences(texts_body2_test)

    # padding操作，将数组全部填充到相同的长度
    s1_train_sequences_pad = sequence.pad_sequences(s1_train_sequence, maxlen=max_sentence_length)
    s2_train_sequences_pad = sequence.pad_sequences(s2_train_sequence, maxlen=max_sentence_length)
    s1_test_sequences_pad = sequence.pad_sequences(s1_test_sequence, maxlen=max_sentence_length)
    s2_test_sequences_pad = sequence.pad_sequences(s2_test_sequence, maxlen=max_sentence_length)

    body1_train_sequences_pad = sequence.pad_sequences(body1_train_sequence, maxlen=max_body_length)
    body2_train_sequences_pad = sequence.pad_sequences(body2_train_sequence, maxlen=max_body_length)
    body1_test_sequences_pad = sequence.pad_sequences(body1_test_sequence, maxlen=max_body_length)
    body2_test_sequences_pad = sequence.pad_sequences(body2_test_sequence, maxlen=max_body_length)



    # # 存储
    # save(embedding_catalog + '/s1_{}_sequences_pad.pickle'.format(train_filename), s1_train_sequences_pad)
    # save(embedding_catalog + '/s2_{}_sequences_pad.pickle'.format(train_filename), s2_train_sequences_pad)
    # save(embedding_catalog + '/s1_{}_sequences_pad.pickle'.format(test_filename), s1_test_sequences_pad)
    # save(embedding_catalog + '/s2_{}_sequences_pad.pickle'.format(test_filename), s2_test_sequences_pad)

    save(embedding_catalog + '/body1_{}_sequences_pad_{}.pickle'.format(train_filename,max_body_length), body1_train_sequences_pad)
    save(embedding_catalog + '/body2_{}_sequences_pad_{}.pickle'.format(train_filename,max_body_length), body2_train_sequences_pad)
    save(embedding_catalog + '/body1_{}_sequences_pad_{}.pickle'.format(test_filename,max_body_length), body1_test_sequences_pad)
    save(embedding_catalog + '/body2_{}_sequences_pad_{}.pickle'.format(test_filename,max_body_length), body2_test_sequences_pad)

    print('结束上述操作，并存储')



if __name__ == '__main__':
    # 加载所有的词汇表训练集和测试集
    pre_deal_train_df = pd.read_csv(train_original,
                                    names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                                           "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig',
                                    sep='\t', lineterminator="\n")
    pre_deal_train_df.fillna('33', inplace=True)  # title经过去除数字的预处理 产生了3条缺失值，我随机填充成数字333
    print(pre_deal_train_df.isnull().any())  # 检查是否有缺失值
    print(pre_deal_train_df[pre_deal_train_df['s1_title'].isnull().values == True])
    print(pre_deal_train_df[pre_deal_train_df['s2_title'].isnull().values == True])

    ################################################################
    test_original = 'data/2018/train/test.csv'
    pre_deal_test_df = pd.read_csv(test_original,
                                   names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title",
                                          "s1_body", "s2_body", "maintag"], header=None, encoding='utf-8-sig', sep='\t',
                                   lineterminator="\n")
    pre_deal_test_df.fillna('33', inplace=True)
    print(pre_deal_test_df.isnull().any())  # 检查是否有缺失值
    print(pre_deal_test_df[pre_deal_test_df['s1_title'].isnull().values == True])
    print(pre_deal_test_df[pre_deal_test_df['s2_title'].isnull().values == True])
    embedding_word(os.path.join('data/2018/embedding', 'word_embedding_matrix.pickle'), pre_deal_train_df,pre_deal_test_df,"train","test")






