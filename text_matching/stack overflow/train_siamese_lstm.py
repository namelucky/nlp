# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from keras.callbacks import TensorBoard

from keras.activations import softmax

from Siamese_LSTM import *
import numpy as np

import pickle
from keras.utils import multi_gpu_model
import tensorflow as tf


embedding_data_catalog = 'data/2018/embedding/'
embedding_body_catalog= 'data/2018/embedding_body/'




def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)


def train_deep_model():
    # 前期参数设置
    embedding_matrix_file_path = 'word_embedding_matrix.pickle'


    RANOD_SEED = 42
    np.random.seed(RANOD_SEED)
    nepoch = 25
    num_folds = 3
    batch_size = 128

    # 加载Embeding矩阵

    embedding_char_matrix=load(embedding_data_catalog + embedding_matrix_file_path)



    y_train = np.array(load("data/2018/train/" + 'y_train.pickle'))



    X_train_s1 = load(embedding_body_catalog + 's1_train_sequences_pad.pickle')
    X_train_s2= load(embedding_body_catalog + 's2_train_sequences_pad.pickle')




    y_train_binary = to_categorical(y_train, 2)  # 简单来说，to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示

    model_checkpoint_path = "model_file/2018/" + 'siamese_lstm_model.h5'
    model =   create_siamese_lstm_ManDistance_model(embedding_char_matrix)
    #    shutil.rmtree('logs')#删除logs目录  为了让logs目录里面只剩下当前运行的结果，
    tensorboard = TensorBoard(log_dir='./logs',  # log 目录
                              histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                              #                  batch_size=32,     # 用多大量的数据计算直方图
                              write_graph=True,  # 是否存储网络结构图
                              write_grads=False,  # 是否可视化梯度直方图
                              write_images=False,  # 是否可视化参数
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None,
                              update_freq='epoch')

    model.fit(x=[X_train_s1, X_train_s2], y=y_train_binary,
                        validation_split=0.1,#0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。
                        batch_size=batch_size,
                        epochs=nepoch,
                        shuffle=True,
                        # batch_size=64, epochs=100,
                        # verbose=1,
                        # class_weight={0: 1.35, 1: 1},
                        # class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。该参数在处理非平衡的训练数据（某些类的训练样本数很少）时，可以使得损失函数对样本数不足的数据更加关注。
                        callbacks=[
                            EarlyStopping(  # 如果监控的目标在设定轮数内不再改善，可以用EarlyStopping回调函数来中断训练
                                monitor='val_loss',
                                # min_delta=0.005,
                                patience=5,
                                verbose=1,
                                mode='auto'
                            ),
                            ModelCheckpoint(  # 作用：按固定间隔，存储模型，默认是每个epoch存储一次模型。
                                model_checkpoint_path,
                                monitor='val_loss',
                                save_best_only=True,  # 只保留最好的检查点,若为True，到固定间隔时，监测值有改进才会保存模型，否则每次固定间隔都会保存一次模型
                                save_weights_only=False,  # 若为True，只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                verbose=1  # 显示信息详细程度。0为不显示，1为显示储存信息
                            ), tensorboard]
                        )

    # plot_model(model, to_file=r"model.png", show_shapes=True, show_layer_names=False)

def mypredict():


    X_test_s1_char = load(embedding_body_catalog + 's1_test_sequences_pad.pickle')
    print(type(X_test_s1_char))
    print(X_test_s1_char.shape)
    X_test_s2_char = load(embedding_body_catalog + 's2_test_sequences_pad.pickle')
    # X_train_s1_char =load(embedding_data_catalog + 'char_s1_train_sequences_pad.pickle')
    # X_train_s2_char = load(embedding_data_catalog + 'char_s2_train_sequences_pad.pickle')
    flag=0
    y_test = np.array(load("data/2018/train/" + 'y_test.pickle'))
    for i in range(0,len(y_test)):
        if y_test[i]==1:
            flag=flag+1

    print(flag)
    # y_test=list(map(int, y_test))
    print(len(y_test))
    # y_test_binary = to_categorical(y_test, 2)  # 0就是[0 0] 1是[0 1]

    model_path = "./model_file/2018/" + 'siamese_lstm_model.h5'
    model0 = load_model(model_path, custom_objects={
                                                     'ManDist':ManDist,
                                                    # 'loss':loss,
                                                    })  # 在Keras中，如果存在自定义layer或者loss，需要在load_model()中以字典形式指定layer或loss，在这里我们就自定义了Mandist层
    print('进行预测操作')
    y_test_predict = predict(model0,  X_test_s1_char,X_test_s2_char)

    y_true = y_test
    print(y_true)
    y_predict = np.argmax(y_test_predict, axis=1)


    print("使用sklearn计算的测试集f1值为{}".format(f1_score(y_true, y_predict)))
    print("使用sklearn计算的测试集准确率为{}".format(accuracy_score(y_true, y_predict)))
    print("使用sklearn计算的测试集精确率为{}".format(precision_score(y_true, y_predict, average='binary')))
    print("使用sklearn计算的测试集召回率为{}".format(recall_score(y_true, y_predict, average='binary')))
    print("使用sklearn计算的测试集auc为{}".format(roc_auc_score(y_true, y_predict)))
    precision = precision_score(y_true, y_predict, average='binary')  # 精确率
    recall = recall_score(y_true, y_predict, average='binary')  # 召回率
    f2_score = 5 * (precision * recall) / (4 * precision + recall)
    print('f2-score为:{}'.format(f2_score))


if __name__ == '__main__':
    train_deep_model()

    mypredict()





