import pandas as pd
import os
from gensim.models import word2vec
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
preprocess_catalog='preprocessing'
from sklearn.model_selection import train_test_split
import jieba

def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # unit
    text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kgs => 4 kg
    text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)  # e.g. 4kg => 4 kg
    text = re.sub(r"(\d+)k ", lambda m: m.group(1) + '000 ', text)  # e.g. 4k => 4000
    text = re.sub(r"\$(\d+)", lambda m: m.group(1) + ' dollar ', text)
    text = re.sub(r"(\d+)\$", lambda m: m.group(1) + ' dollar ', text)

    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"c\+\+", "cplusplus", text)
    text = re.sub(r"c \+\+", "cplusplus", text)
    text = re.sub(r"c \+ \+", "cplusplus", text)
    text = re.sub(r"c#", "csharp", text)
    text = re.sub(r"f#", "fsharp", text)
    text = re.sub(r"g#", "gsharp", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r",000", '000', text)
    text = re.sub(r"\'s", " ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r"pokemons", "pokemon", text)
    text = re.sub(r"pokémon", "pokemon", text)
    text = re.sub(r"pokemon go ", "pokemon-go ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" 9 11 ", " 911 ", text)
    text = re.sub(r" j k ", " jk ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r"insidefacebook", "inside facebook", text)
    text = re.sub(r"donald trump", "trump", text)
    text = re.sub(r"the big bang", "big-bang", text)
    text = re.sub(r"the european union", "eu", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" quaro ", " quora ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r" rs(\d+)", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"(\d+)rs", lambda m: ' rs ' + m.group(1), text)
    text = re.sub(r"the european union", " eu ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"₹", " rs ", text)  # 测试！
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text

def train_word_w2v(pre_deal_train_df,f_name,binary = False):
    """
    训练词向量
    :param file_name:需要进行训练词向量的文本文件名字，位于preprocessing目录中
    :param f_name:生成的词向量的文件名
    :param binary:将词向量表是否存储为二进制文件
    :return:
    """
    # 加载所有的词汇表训练集和测试集
    #pre_deal_train_df =pd.read_csv("data/all.csv", names=['id', 'qid1', 'qid2', 'question1', 'question2','is_duplicate'], sep=',', header=0,encoding='utf-8-sig')
    pre_deal_train_df.fillna('aa',inplace=True)
    print(pre_deal_train_df.info())
    texts = []
    texts_s1_train = [line.strip().split(" ") for line in pre_deal_train_df['question1'].tolist()]
    texts_s2_train = [line.strip().split(" ") for line in pre_deal_train_df['question2'].tolist()]
    texts.extend(texts_s1_train)#extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    texts.extend(texts_s2_train)

    model = word2vec.Word2Vec(sentences=texts,size=300,window=2,min_count=2,workers=2)
    #sentences：可以是一个list，是训练预料，min_count是小于该数的单词会被踢出，默认值为5，window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少， workers：用于控制训练的并行数
    model.wv.save_word2vec_format(fname=f_name,binary=binary,fvocab=None)

def preprocessing(data_df,fname):
    """
    :param data_df:需要处理的数据集
    :param fname:
    :return:生成的预处理文件在preprocessing文件中
    """
    for index, row in data_df.iterrows():
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ["question1", "question2"]:

            clean_str=re.sub(r"[^a-zA-Z ']", "", str(row[col_name]))##只保留a-z A-Z 0-9 其余用空格代替
            out_str = clean_str.strip()#去除开头和结尾的空格
            out_str=out_str.lower()##将文本统一转换大小写，统一小写
            out_str=clean_text(out_str)
            seg_str = seg_sentence(str(out_str))
            #print(type(out_str))
            # #去除停用词
            # out_str=[word for word in seg_str if word not in stopwords.words("english")]
            #词形还原Lemmatization 例如将started还原成start
            out_str=[WordNetLemmatizer().lemmatize(word) for word in seg_str]


            data_df.at[index, col_name] = out_str

    data_df.to_csv('preprocessing/preprocessed_' + '{}.csv'.format(fname), sep='\t', header=None,index=None,encoding='utf-8')

def seg_sentence(sentence):
    """
    对句子进行分词
    :param sentence:句子，String
    """
    sentence_seged = jieba.cut(sentence.strip())#strip()方法，去除字符串开头或者结尾的空格
    out_str = ""
    for word in sentence_seged:
        if word != " ":
            out_str += word
            out_str += " "
    out_str=out_str.strip()
    return out_str

def stat(df):
    '''
    获取句子长度分布和数据集正负样本分布
    :param df:
    :return:
    '''
    print(df.isnull().any())#检查是否有缺失值
    print(df.info())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    df['q1_n_words'] = df['question1'].apply(lambda row: len(row.split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda row: len(row.split(" ")))
    sns.distplot(df['q1_n_words'],ax=ax1,label='question1',color='blue')
    sns.distplot(df['q2_n_words'],ax=ax2,label='question2',color='green')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax1.set_title('数据集question1长度分布')
    ax2.set_title('数据集question2长度分布')
    plt.show()
    a=df.groupby(['is_duplicate'])['id'].count().plot.bar()
    a.set_title("数据集正负样本分布")
    plt.show()

def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)

def divide(data_df):

    X_df=data_df[['id', 'qid1', 'qid2', 'question1', 'question2']]
    Y_df=data_df[['is_duplicate']]
    X=X_df.values.tolist()
    Y=Y_df.values.tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

    X_train_df=pd.DataFrame(columns=['id', 'qid1', 'qid2', 'question1', 'question2'],index=None,data=X_train)
    X_test_df = pd.DataFrame(columns=['id', 'qid1', 'qid2', 'question1', 'question2'],index=None, data=X_test)
    Y_train_df=pd.DataFrame(columns=['is_duplicate'],data=Y_train)
    Y_test_df = pd.DataFrame(columns=['is_duplicate'], data=Y_test)


    X_train_df.insert(5, 'is_duplicate', Y_train_df)
    X_train_df.to_csv('./data/train.csv',encoding='utf-8',header=None,index=None, sep='\t')


    X_test_df.insert(5, 'is_duplicate', Y_test_df)

    X_test_df.to_csv('./data/test.csv',encoding='utf-8',header=None,index=None, sep='\t')



if __name__ == '__main__':
    train_df=pd.read_csv("data/train.csv", names=['id', 'qid1', 'qid2', 'question1', 'question2','is_duplicate'], sep='\t', header=None,encoding='utf-8-sig')
    # #divide(train_df)
    test_df = pd.read_csv("data/test.csv", names=['id', 'qid1', 'qid2', 'question1', 'question2','is_duplicate'],sep='\t', header=None, encoding='utf-8-sig')
    preprocessing(train_df,'train')#数据预处理，去除标点符号
    preprocessing(test_df,'test')
    save('preprocessing/train_y.pickle',train_df['is_duplicate'].tolist())
    save('preprocessing/test_y.pickle', test_df['is_duplicate'].tolist())
    # # #

    all_df = pd.read_csv("data/all.csv", names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                           sep=',', header=0, encoding='utf-8-sig')
    preprocessing(all_df, 'all')
    p_df=pd.read_csv('./preprocessing/preprocessed_all.csv', names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                           sep='\t', header=None, encoding='utf-8-sig')
    # # print(p_df.info())
    # # print(p_df.head())
    train_word_w2v(p_df,'w2v.bigram')#训练字向量
    # stat(train_df)#获取passage和query长度分布
    #
    train_df=pd.read_csv('./preprocessing/preprocessed_train.csv', names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                           sep='\t', header=None, encoding='utf-8-sig')
    train_df.fillna('ok',inplace=True)
    print(train_df.info())
    train_df.to_csv('preprocessing/preprocessed_' + '{}.csv'.format('train'), sep='\t', header=None, index=None,
                   encoding='utf-8')

    test_df=pd.read_csv('./preprocessing/preprocessed_test.csv', names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                           sep='\t', header=None, encoding='utf-8-sig')
    test_df.fillna('ok',inplace=True)
    print(test_df.info())
    test_df.to_csv('preprocessing/preprocessed_' + '{}.csv'.format('test'), sep='\t', header=None, index=None,
                   encoding='utf-8')

    #

