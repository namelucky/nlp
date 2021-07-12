#
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import logging

logging.basicConfig(level=logging.ERROR)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import f1_score,accuracy_score
from transformers import *
from tensorflow.keras.layers import Dense, GlobalAveragePooling1D, BatchNormalization
import random
from tensorflow.keras.metrics import AUC, Precision, Recall
from keras.layers import *
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
def load( nfile):
    with open(nfile, 'rb') as file:
        return pickle.load(file)

y_true=np.array(load("./bert_input/" + 'test_label.pickle'))
print(np.array(y_true).shape)



PATH =  r'/home/znuser2/lwl/pretrained_model/en/bert_h5/'
BERT_PATH = PATH
BERT_PATH = PATH
MAX_SEQUENCE_LENGTH = 20


def create_model():
    q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

    config = BertConfig.from_pretrained(
        BERT_PATH +'config.json')
    config.output_hidden_states = False
    bert_model = TFBertModel.from_pretrained(
        BERT_PATH +'bert.h5',
        config=config)



    abs_layer = tf.keras.layers.Lambda(lambda x: K.abs(x))
    q_nsp = bert_model(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_nsp = bert_model(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]


    print('q_np',q_nsp)
    q_nsp_pooling=GlobalAveragePooling1D()(q_nsp)
    a_nsp_pooling = GlobalAveragePooling1D()(a_nsp)
    print('q_nsp_pooling',q_nsp_pooling)
    print(a_nsp_pooling)
    subtracted = tf.keras.layers.Subtract()([q_nsp_pooling, a_nsp_pooling])
    abs_output = abs_layer(subtracted)
    x = Concatenate()([q_nsp_pooling, a_nsp_pooling,abs_output])


    print(x)
    y = Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, a_id, a_mask, a_atn], outputs=y)
    return model






train_inputs_q = np.load('./bert_input/inputs_q.npy',)
train_inputs_a = np.load('./bert_input/inputs_a.npy',)
train_outputs = np.load('./bert_input/outputs.npy')
print(train_inputs_q)


test_inputs_q = np.load('./bert_input/test_inputs_q.npy')
test_inputs_a = np.load('./bert_input/test_inputs_a.npy')
test_outputs = np.load('./bert_input/test_outputs.npy')


test_preds = []


model = create_model()

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[AUC(), Precision(), Recall()])
#model.load_weights('model/final_ernie_01_{0:d}.h5'.format(int(i)))




train_start_index=0

train_end_index=len(train_outputs)

print(train_start_index,train_end_index)



train_inputs_q = [i[train_start_index:train_end_index] for i in train_inputs_q]
train_inputs_a =[i[train_start_index:train_end_index] for i in train_inputs_a]
train_outputs = train_outputs[train_start_index:train_end_index]


test_start_index=0

test_end_index=len(test_outputs)
test_inputs_q = [i[test_start_index:test_end_index] for i in test_inputs_q]
test_inputs_a = [i[test_start_index:test_end_index] for i in test_inputs_a]




model_checkpoint_path='./save_model/sent_bert_01.h5'
tensorboard=TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=False,embeddings_freq=0,update_freq='epoch')

train_start=time.perf_counter()
model.fit([train_inputs_q,train_inputs_a], train_outputs, validation_split=0.1,
          epochs=1,
          batch_size=32,
          callbacks=[
              EarlyStopping(  # 如果监控的目标在设定轮数内不再改善，可以用EarlyStopping回调函数来中断训练
                  monitor='val_loss',
                  min_delta=0.005,
                  patience=4,
                  verbose=1,
                  mode='auto'
              ),
              ModelCheckpoint(  # 作用：按固定间隔，存储模型，默认是每个epoch存储一次模型。
                  model_checkpoint_path,
                  monitor='val_loss',
                  save_best_only=True,  # 只保留最好的检查点,若为True，到固定间隔时，监测值有改进才会保存模型，否则每次固定间隔都会保存一次模型
                  save_weights_only=True,  # 若为True，只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                  verbose=1  # 显示信息详细程度。0为不显示，1为显示储存信息
              )]

          )
train_end=time.perf_counter()

# model.save_weights('./save_model/sent_bert_01.h5')



start=time.perf_counter()
test_preds.append(model.predict([test_inputs_q, test_inputs_a], batch_size=512))
end=time.perf_counter()

sub = np.average(test_preds, axis=0)
prob = sub
sub = sub > 0.5
# df_test['prob'] = prob
# df_test['label_pre'] = sub.astype(int)
y_predict=sub.astype(int)
y_true=np.array(load("./bert_input/" + 'test_label.pickle'))
# df_test['label_pre'].to_csv('./result.tsv', index=False, header=None,sep='\t')

print("使用sklearn计算的测试集f1值为{}".format(f1_score(y_true, y_predict)))
print("使用sklearn计算的测试集准确率为{}".format(accuracy_score(y_true, y_predict)))
print("使用sklearn计算的测试集精确率为{}".format(precision_score(y_true, y_predict, average='binary')))
print("使用sklearn计算的测试集召回率为{}".format(recall_score(y_true, y_predict, average='binary')))
print("使用sklearn计算的测试集auc为{}".format(roc_auc_score(y_true, y_predict)))
precision = precision_score(y_true, y_predict, average='binary')  # 精确率
recall = recall_score(y_true, y_predict, average='binary')  # 召回率
f2_score = 5 * (precision * recall) / (4 * precision + recall)
print('f2-score为:{}'.format(f2_score))

logging.INFO("测试用的时间为{}",end-start)
logging.INFO("用的时间为{}",train_end-train_start)
