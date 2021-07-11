# coding=utf-8

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from keras.callbacks import TensorBoard
import os
from keras.activations import softmax
from model_siamese_body import *
# from siamese_lstm import *
#  from DCNN import *
import pickle
from keras.utils import multi_gpu_model
import tensorflow as tf


embedding_data_catalog = 'embedding/'
embedding_body_catalog = 'embedding/'


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)

def mypredict(file):


    X_test_s1_char = load(embedding_body_catalog + 's1_{}_sequences_pad.pickle'.format(file))

    X_test_s2_char = load(embedding_body_catalog + 's2_{}_sequences_pad.pickle'.format(file))

    X_test_body1_char = load(embedding_body_catalog + 'body1_{}_sequences_pad.pickle'.format(file))
    X_test_body2_char = load(embedding_body_catalog + 'body2_{}_sequences_pad.pickle'.format(file))


    flag=0
    y_test = np.array(load('y_{}.pickle'.format(file)))
    for i in range(0,len(y_test)):
        if y_test[i]==1:
            flag=flag+1

    print(flag)
    # y_test=list(map(int, y_test))
    print(len(y_test))
    # y_test_binary = to_categorical(y_test, 2)  # 0就是[0 0] 1是[0 1]

    model_path = '../model_file/2018/model_siamese_body_without_re.h5'
    model0 = load_model(model_path, custom_objects={'F1': F1, 'Precision': Precision,
                                                    'Recall': Recall,
                                                    'submult': submult,
                                                    'softmax': softmax,
                                                     # 'ManDist':ManDist,
                                                    # 'loss':loss,
                                                    })  # 在Keras中，如果存在自定义layer或者loss，需要在load_model()中以字典形式指定layer或loss，在这里我们就自定义了Mandist层
    print('进行预测操作')
    begin=time.perf_counter()
    y_test_predict = predict(model0,  X_test_s1_char,X_test_s2_char, X_test_body1_char,X_test_body2_char)
    end = time.perf_counter()

    # 将测试集y 由one-hot形式重新变为十进制形式 [1 0]->0  [0 1] ->1
    y_true = y_test
    print(y_true)

    y_predict = np.argmax(y_test_predict, axis=1)

    print('time used', end - begin)
    # save('temp/y_true.pickle', y_true)
    # save('temp/y_predict.pickle', y_predict)

    print("使用sklearn计算的测试集f1值为{}".format(f1_score(y_true, y_predict)))
    print("使用sklearn计算的测试集准确率为{}".format(accuracy_score(y_true, y_predict)))
    print("使用sklearn计算的测试集精确率为{}".format(precision_score(y_true, y_predict, average='binary')))
    print("使用sklearn计算的测试集召回率为{}".format(recall_score(y_true, y_predict, average='binary')))
    print("使用sklearn计算的测试集auc为{}".format(roc_auc_score(y_true, y_predict)))
    precision = precision_score(y_true, y_predict, average='binary')  # 精确率
    recall = recall_score(y_true, y_predict, average='binary')  # 召回率
    f2_score = 5 * (precision * recall) / (4 * precision + recall)
    print('f2-score为:{}'.format(f2_score))

import time
if __name__ == '__main__':

    mypredict("mysql")





