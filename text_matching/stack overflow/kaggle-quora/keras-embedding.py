import pandas as pd
from gensim.models import KeyedVectors
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import pickle


preprocess_catalog='preprocessing'
embedding_catalog='embedding_data'#做完embedding操作，文件存放的文件夹
embedding_size = 300#词向量的维度
max_sentence_length = 30#padding的长度
max_vovab_size = 100000#tokenizer的numword


def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)

def process_save_embedding_word(nfile):
    """
    :param type:
    :return:
    """

    w2v_path ='w2v.bigram'
    print ('load w2v_model...')
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    print ('finish w2v_model...')

    tokenizer = Tokenizer(
        num_words=max_vovab_size,
        split=' ',
        lower=False,
        char_level=False,
        filters=''
    )
    # 加载训练集和测试集
    pre_train_df = pd.read_csv(os.path.join(preprocess_catalog, 'preprocessed_train.csv'),
                                    names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                                    header=None,encoding='utf-8',
                                    sep='\t')
    print(pre_train_df.info())
    print(pre_train_df.head())
    pre_test_df = pd.read_csv(os.path.join(preprocess_catalog,'preprocessed_test.csv'),
                                   names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                                   header=None,encoding='utf-8',
                                   sep='\t',
                                   )
    print(pre_test_df.info())
    print(pre_test_df.head())
    texts = []
    texts_queries_test = pre_test_df['question1'].tolist()
    texts_passage_test = pre_test_df['question2'].tolist()

    texts_queries_train = pre_train_df['question1'].tolist()
    texts_passage_train = pre_train_df['question2'].tolist()

    texts.extend(texts_queries_test )
    texts.extend(texts_passage_test)
    texts.extend(texts_queries_train)
    texts.extend(texts_passage_train)

    tokenizer.fit_on_texts(texts)
    num_words_dict = tokenizer.word_index

    # 训练集的词汇表的词向量矩阵,行数为最大值+1,列数为词向量的维度，形式为：index->vec，主要是为了构建单词在字典中的序号index->单词的词向量
    embedding_matrix = 1 * np.random.randn(len(num_words_dict) + 1, embedding_size)
    print('keras的embedding矩阵维度为[{},{}]'.format(len(num_words_dict) + 1,embedding_size))
    embedding_matrix[0] = np.random.randn(embedding_size)

    count = 0
    nocount = 0
    for word,index in num_words_dict.items():#index是从1开始的
        if word in w2v_model.vocab:
            embedding_matrix[index] = w2v_model.word_vec(word)
            count = count +1
        else:
            print('{}没有进行向量化'.format(word))
            nocount=nocount+1;
    print('total {}, word in model have {}'.format(len(num_words_dict),count))
    print('有{}个词没有进行embedding向量化'.format(nocount))
    save(nfile,embedding_matrix)
    print('finish')
    print('将对应文本转为数字，并进行padding操作')
    # 将文本向量化，将单词转为对应的字典的数字  ['ha ha gua angry'] {{'ha': 1, 'gua': 2, 'angry': 3,.....},文本就会变成[1, 1, 2, 3],
    #s1指代的就是query s2指代的就是passage
    s1_train_sequence = tokenizer.texts_to_sequences(texts_queries_train)
    s2_train_sequence = tokenizer.texts_to_sequences(texts_passage_train)
    s1_test_sequence = tokenizer.texts_to_sequences(texts_queries_test)
    s2_test_sequence = tokenizer.texts_to_sequences(texts_passage_test)

    #padding操作，将数组全部填充到相同的长度
    s1_train_sequences_pad = sequence.pad_sequences(s1_train_sequence,maxlen=max_sentence_length)
    s2_train_sequences_pad = sequence.pad_sequences(s2_train_sequence,maxlen=max_sentence_length)
    s1_test_sequences_pad = sequence.pad_sequences(s1_test_sequence,maxlen=max_sentence_length)
    s2_test_sequences_pad = sequence.pad_sequences(s2_test_sequence,maxlen=max_sentence_length)
    #存储
    save(embedding_catalog+ '/s1_train_sequences_pad.pickle', s1_train_sequences_pad)
    save(embedding_catalog + '/s2_train_sequences_pad.pickle', s2_train_sequences_pad)
    save(embedding_catalog + '/s1_test_sequences_pad.pickle', s1_test_sequences_pad)
    save(embedding_catalog + '/s2_test_sequences_pad.pickle', s2_test_sequences_pad)
    print('结束上述操作，并存储')


if __name__ == '__main__':
    process_save_embedding_word('embedding_data/word_embedding_matrix.pickle')

