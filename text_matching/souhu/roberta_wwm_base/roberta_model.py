# -*-coding=utf-8-*-
'''
@Created on 2021/4/8


'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,1'
import logging

logging.basicConfig(level=logging.ERROR)
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import f1_score
from transformers import *
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.metrics import AUC, Precision, Recall

import random
from transformers import RobertaTokenizer, TFRobertaModel
import logging
from loadlog import configure_logging
from keras.models import load_model
configure_logging("logging_config.json")

# MODEL_PATH = r'/root/lwl/pretrained-model/roberta/'

MODEL_PATH =  r'nghuyong/ernie-1.0'

def load_data(name=None):
    all_train_input = np.load('data/all_train_input.npy', )
    all_train_input_type = np.load('data/all_train_input_type.npy', )
    all_train_label = np.load('data/all_train_label.npy', )

    all_test_input = np.load('data/all_test_input.npy', )
    all_test_input_type = np.load('data/all_test_input_type.npy')
    all_test_id = np.load('data/all_test_id.npy')

    train_input = [all_train_input[0], all_train_input[1], all_train_input[2], all_train_input_type]
    test_input = [all_test_input[0], all_test_input[1], all_test_input[2], all_test_input_type]

    return train_input, all_train_label, test_input, all_test_id




def create_model(MAX_SEQUENCE_LENGTH):
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    # config=BertConfig.from_pretrained(MODEL_PATH+"config.json")

    roberta_model = TFBertModel.from_pretrained(MODEL_PATH)
    roberta_model.trainable = True

    roberta = roberta_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    q = roberta[:, 0]

    type = tf.keras.layers.Input(shape=(1,))
    # 2种类型
    c = tf.keras.layers.Embedding(2, int(q.shape[-1]))(type)
    c = tf.squeeze(c, axis=1)
    # logging.info("c的维度是{}".format(c))
    # logging.info("q的维度是{}".format(q))
    q = tf.keras.layers.Average()([q, c])
    # logging.info("连接后的维度是{}".format(q))
    # a=tf.keras.layers.MaxPooling1D()([q, c])
    y = Dense(1, activation='sigmoid')(q)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, type], outputs=y)

    model.summary()
    return model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def search_f1(y_true, y_pred):
    best = 0
    best_t = 0
    for i in range(30, 50):
        tres = i / 100.0
        y_pred_bin = (y_pred > tres).astype(int)
        score = f1_score(y_true, y_pred_bin)
        if score > best:
            best = score
            best_t = tres
    print('best', best)
    print('thres', best_t)
    return best, best_t


def train(model_name, max_len):
    # 加载处理后的数据
    input, lable, test_input, test_id = load_data()

    # 只打乱训练集
    index = [i for i in range(len(lable))]
    random.shuffle(index)
    lable = lable[index]
    input = [i[index] for i in input]

    # 5折交叉
    k_flod = 5
    num = len(lable) // k_flod

    for i in range(k_flod):
        X_train = input
        y_train = lable

        # 划分训练集和验证集
        val_start_index = int(i * num)
        val_end_index = val_start_index + num
        # print(val_start_index, val_end_index)

        valid_inputs = [i[val_start_index:val_end_index] for i in X_train]
        valid_outputs = y_train[val_start_index:val_end_index]

        train_inputs = [np.concatenate([i[:val_start_index], i[val_end_index:]], axis=0) for i in X_train]
        train_outputs = np.concatenate([y_train[:val_start_index], y_train[val_end_index:]], axis=0)

        K.clear_session()
        # strategy = tf.distribute.MirroredStrategy(
        #     devices=["/device:GPU:0", "/device:GPU:1"])
        # with strategy.scope():
        model = create_model(max_len)

        # 设置学习率
        def scheduler(epoch):
            # 每3次更新一次学习率
            interval_epoch = 2
            if epoch % interval_epoch == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr / 10)
                print("lr changed to {}".format(lr / 10))
            return K.get_value(model.optimizer.lr)

        # 设置动态学习率
        reduce_lr = LearningRateScheduler(scheduler)
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        # model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], optimizer=optimizer,
        #               metrics=[f1, Precision(), Recall()])
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f1, Precision(), Recall()])

        model.fit(train_inputs, train_outputs, validation_data=(valid_inputs, valid_outputs),
                  epochs=2,
                  batch_size=8,
                  # callbacks=[reduce_lr],
                  # class_weight={0: 1., 1: 0.95},
                  )
        #我没有保存整个模型，我只保存了模型的权重，如果保存整个模型 是 model.save()
        model.save_weights('model/{0:s}_{1:d}.h5'.format(model_name, int(i)))


def predict(model_name, max_len):
    input, lable, test_input, test_id = load_data()

    # 只打乱训练集
    index = [i for i in range(len(lable))]
    random.shuffle(index)
    lable = lable[index]
    input = [i[index] for i in input]

    # 5折交叉
    k_flod = 5
    num = len(lable) // k_flod

    test_preds = []
    train_preds = []
    valid_preds = []
    best_score, best_t = 0, 0

    for i in range(k_flod):

        X_train = input
        y_train = lable

        # 划分训练集和验证集
        val_start_index = int(i * num)
        val_end_index = val_start_index + num


        valid_inputs = [i[val_start_index:val_end_index] for i in X_train]
        valid_outputs = y_train[val_start_index:val_end_index]
        train_inputs = [np.concatenate([i[:val_start_index], i[val_end_index:]], axis=0) for i in X_train]
        train_outputs = np.concatenate([y_train[:val_start_index], y_train[val_end_index:]], axis=0)

        logging.info("train_outs维度 {}".format(train_outputs.shape))



        K.clear_session()
        strategy = tf.distribute.MirroredStrategy(
            devices=["/device:GPU:2","/device:GPU:1"])
        with strategy.scope():
            model = create_model(max_len)
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[f1, Precision(), Recall()])
            model.load_weights('model/{0:s}_{1:d}.h5'.format(model_name, int(i)))

            # 预测验证集
            valid_pred = model.predict([i for i in valid_inputs], batch_size=128)
            train_pred=  model.predict([i for i in train_inputs], batch_size=128)

            valid_preds.append(valid_pred)
            logging.info("valid_preds维度 {}".format(np.array(valid_preds).shape))
            train_preds.append(train_pred)
            logging.info("train_preds维度 {}".format(np.array(train_preds).shape))
            score, t = search_f1(valid_outputs, valid_pred)

            logging.info("pair model K-{}, validation score ={}".format(i, score))
            # 选择最佳的阈值
            if score > best_score:
                best_score, best_t = score, t
            logging.info("第{}次的验证集best_score{},最佳阈值{}".format(i,best_score,best_t))

            # 预测测试集
            test_preds.append(model.predict([i for i in test_input], batch_size=16))
            # if i==1:
            #     break
        # if i==2:
        #     continue
    # np.save('test_preds.npy', test_preds)

    # 将5个模型预测的结果进行平均
    sub = np.average(test_preds, axis=0)

    sub_train=np.average(train_preds, axis=0)
    sub_valid = np.average(valid_preds, axis=0)

    train_preds = np.squeeze(train_preds).T#squeeze是为了去除维度为1的列，train_preds是（5, 55663, 1）维度的，sub_train是(55663,)，通过变换，将train_preds转为(55663,5)
    valid_preds = np.squeeze(valid_preds).T

    # print("train_preds维度 {}".format(np.array(train_preds).shape))
    train_preds = np.concatenate((train_preds, sub_train), axis=1)
    valid_preds = np.concatenate((valid_preds, sub_valid), axis=1)#将(13915,5)和(13915,1)合并 ->(13915,6)

    train_outputs = np.expand_dims(train_outputs, axis=1)#train_outputs维度原先是(55663,) 通过升维，变为(55663,1)
    valid_outputs = np.expand_dims(valid_outputs, axis=1)

    train_preds = np.concatenate((train_preds, train_outputs), axis=1)
    valid_preds = np.concatenate((valid_preds, valid_outputs), axis=1)

    # 真正完整的训练数据，之所以开始的时候分开做了训练和验证，是为了单独在验证集中找最好的阈值，给测试集确定label

    Train = np.concatenate((train_preds, valid_preds), axis=0).tolist()#axis=0表示在纵轴上合并，axis=1表示在横轴上合并

    Train_df=pd.DataFrame(Train,columns=['K0','K1','K2','K3','K4',"mean","label"])
    logging.info(Train_df)
    Train_df.to_csv('res/{}_train_prob.csv'.format(model_name), index=False,
                                             header=None, sep=',', mode='w')



    # 预测的概率
    prob = sub
    # 预测标签
    sub = sub > best_t
    pred_lable = sub.astype(int)

    df_test = pd.DataFrame({'id': np.squeeze(test_id),'label': np.squeeze(pred_lable),'prob': np.squeeze(prob)})

    # 保存两份（单独预测一份，汇总结果一份）
    df_test[['id', 'label']].to_csv('res/{0:s}_submission_{1:.4f}.csv'.format(name, best_score), index=False,
                                    header=None, sep=',', mode='w')
    df_test[['id', 'label', 'prob']].to_csv('res/{0:s}_test_prob_{1:.4f}.csv'.format(name, best_score), index=False,
                                             header=None, sep=',', mode='w')
    #
    # df_test[['id', 'label']].to_csv('res/all_submission.csv', index=False, header=['id', 'label'], sep=',', mode='a+')
    # df_test[['id', 'label', 'prob']].to_csv('res/all_test_prob.csv', index=False, header=None, sep=',', mode='a+')

if __name__ == '__main__':
    #train('ernie', 256)
    predict("ernie",256)




