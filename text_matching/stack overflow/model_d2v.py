import numpy as np
import pandas as pd
from keras.layers import *
from keras import layers

from keras.activations import softmax
from keras.models import Model
from keras.models import Model
from keras.utils import plot_model
from keras.layers import *
import keras.backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
import logging
from loadlog import configure_logging
configure_logging("logging_config.json")

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return attention,in1_aligned, in2_aligned



def textcnn1(embedding_matrix,embedding_size = 300,max_sentence_length = 20,max_body_length=30,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.5):


    q1 = Input(name='q1', shape=(max_sentence_length,))
    q2 = Input(name='q2', shape=(max_sentence_length,))

    q3 = Input(name='q3', shape=(max_body_length,))
    q4 = Input(name='q4', shape=(max_body_length,))

    # Embedding
    embedding = Embedding(
        input_dim=len(embedding_matrix, ),  # 整个词典的大小。例如nlp中唯一单词的个数，当然如果还有一个用于padding的单词那就应加1
        output_dim=embedding_size,  # 嵌入维度。例如nlp中设计的词向量的维度
        weights=[embedding_matrix],
        trainable=False,
        input_length=max_sentence_length  # 输入序列的长度
    )
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))
    print('q1_embed',q1_embed)

    q3_embed = bn(embedding(q3))
    q4_embed = bn(embedding(q4))

    # Encode

    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    encode2=Bidirectional(LSTM(lstm_dim, return_sequences=False))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)
    # q5=encode(q1_embed)
    # logging.info(K.squeeze(q5[:, -1:],axis=1))
    print('q1_encoded',q1_encoded)

    q3_encoded = encode(q3_embed)
    # q3_encoded=K.squeeze(q3_encoded[:, -1:],axis=1)

    q4_encoded = encode(q4_embed)

    atten, q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)
    _, q3_aligned, q4_aligned = soft_attention_alignment(q3_encoded, q4_encoded)
    atten = Activation('softmax')(atten)
    a_pool=GlobalAveragePooling1D(name='avg')(atten)
    a_max=GlobalMaxPooling1D(name='max')(atten)
    print(a_pool)

    # Compose
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    q3_combined = Concatenate()(
        [q3_encoded, q4_aligned, submult(q3_encoded, q4_aligned)])
    q4_combined = Concatenate()(
        [q4_encoded, q3_aligned, submult(q4_encoded, q3_aligned)])


    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)
    q4_compare = compose(q4_combined)
    q3_compare = compose(q3_combined)


    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    q3_rep = apply_multiple(q3_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q4_rep = apply_multiple(q4_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    print(q1_rep)
    abs_layer = tf.keras.layers.Lambda(lambda x: K.abs(x))

    dense = Concatenate()([q1_rep, q2_rep,a_max,a_pool,abs_layer(q3-q4),Multiply()([q3, q4])])

    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)

    out_ = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[q1, q2,q3,q4], outputs=out_)

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=[Precision, Recall, F1, ])
    model.summary()
    return model






def Recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def Precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

def predict(model,X_s1_char,X_s2_char,X_body1_char,X_body2_char):

    y1 = model.predict([X_s1_char,X_s2_char,X_body1_char,X_body2_char])

    res =y1
    # print res[0:15]
    return res

def loss(y_true, y_pred):
    return - (K.mean(K.square(y_true - y_pred)) * y_true * K.log(y_pred + 1e-8) + (1-K.mean(K.square(y_true - y_pred))) * (1 - y_true) * K.log(1 - y_pred + 1e-8))