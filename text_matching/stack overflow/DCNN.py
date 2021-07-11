import pickle
import numpy as np
import os
import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import *
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, Reshape
from keras.layers import MaxPooling2D, Dropout
from keras.models import Model
from keras.layers import Layer
import keras
import tensorflow.keras.backend as K
import random
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)


def DCCNN_Model(embedding_matrix,embedding_size = 300,max_sentence_length = 20,):
    q1 = Input(name='q1', shape=(max_sentence_length,))
    q2 = Input(name='q2', shape=(max_sentence_length,))


    # Embedding
    embedding = Embedding(
        input_dim=len(embedding_matrix, ),  # 整个词典的大小。例如nlp中唯一单词的个数，当然如果还有一个用于padding的单词那就应加1
        output_dim=embedding_size,  # 嵌入维度。例如nlp中设计的词向量的维度
        weights=[embedding_matrix],
        trainable=False,
        input_length=max_sentence_length  # 输入序列的长度
    )
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)
    X_input=tf.stack([q1_embed,q2_embed],axis=0)
    X_input=tf.transpose(X_input,perm=[1,3,2,0])
    print('X_input',X_input)


    X_1 = Conv2D(100, kernel_size=(1, 20), strides=(1, 1), activation='relu')(X_input)
    X_1 = layers.normalization.BatchNormalization(axis=-1)(X_1)
    X_1 = Reshape((300, 100, 1))(X_1)

    X_1_1 = Conv2D(200, kernel_size=(1, 100), strides=(1, 1), activation='relu')(X_1)
    X_1_1 = layers.normalization.BatchNormalization(axis=-1)(X_1_1)
    X_1_1 = MaxPooling2D(pool_size=(300, 1), padding='valid')(X_1_1)
    X_1_1 = Flatten()(X_1_1)
    X_1_2 = Conv2D(200, kernel_size=(2, 100), strides=(1, 1), activation='relu')(X_1)
    X_1_2 = layers.normalization.BatchNormalization(axis=-1)(X_1_2)
    X_1_2 = MaxPooling2D(pool_size=(299, 1), padding='valid')(X_1_2)
    X_1_2 = Flatten()(X_1_2)
    X_1_3 = Conv2D(200, kernel_size=(3, 100), strides=(1, 1), activation='relu')(X_1)
    X_1_3 = layers.normalization.BatchNormalization(axis=-1)(X_1_3)
    X_1_3 = MaxPooling2D(pool_size=(298, 1), padding='valid')(X_1_3)
    X_1_3 = Flatten()(X_1_3)

    X_1 = layers.Concatenate(axis=-1)([X_1_1, X_1_2])
    X_1 = layers.Concatenate(axis=-1)([X_1, X_1_3])

    X_2 = Conv2D(100, kernel_size=(2, 20), strides=(1, 1), activation='relu')(X_input)
    X_2 = layers.normalization.BatchNormalization(axis=-1)(X_2)
    X_2 = Reshape((299, 100, 1))(X_2)

    X_2_1 = Conv2D(200, kernel_size=(1, 100), strides=(1, 1), activation='relu')(X_2)
    X_2_1 = layers.normalization.BatchNormalization(axis=-1)(X_2_1)
    X_2_1 = MaxPooling2D(pool_size=(299, 1), padding='valid')(X_2_1)
    X_2_1 = Flatten()(X_2_1)
    X_2_2 = Conv2D(200, kernel_size=(2, 100), strides=(1, 1), activation='relu')(X_2)
    X_2_2 = layers.normalization.BatchNormalization(axis=-1)(X_2_2)
    X_2_2 = MaxPooling2D(pool_size=(298, 1), padding='valid')(X_2_2)
    X_2_2 = Flatten()(X_2_2)
    X_2_3 = Conv2D(200, kernel_size=(3, 100), strides=(1, 1), activation='relu')(X_2)
    X_2_3 = layers.normalization.BatchNormalization(axis=-1)(X_2_3)
    X_2_3 = MaxPooling2D(pool_size=(297, 1), padding='valid')(X_2_3)
    X_2_3 = Flatten()(X_2_3)

    X_2 = layers.Concatenate(axis=-1)([X_2_1, X_2_2])
    X_2 = layers.Concatenate(axis=-1)([X_2, X_2_3])

    X_3 = Conv2D(100, kernel_size=(3, 20), strides=(1, 1), activation='relu')(X_input)
    X_3 = layers.normalization.BatchNormalization(axis=-1)(X_3)
    X_3 = Reshape((298, 100, 1))(X_3)

    X_3_1 = Conv2D(200, kernel_size=(1, 100), strides=(1, 1), activation='relu')(X_3)
    X_3_1 = layers.normalization.BatchNormalization(axis=-1)(X_3_1)
    X_3_1 = MaxPooling2D(pool_size=(298, 1), padding='valid')(X_3_1)
    X_3_1 = Flatten()(X_3_1)
    X_3_2 = Conv2D(200, kernel_size=(2, 100), strides=(1, 1), activation='relu')(X_3)
    X_3_2 = layers.normalization.BatchNormalization(axis=-1)(X_3_2)
    X_3_2 = MaxPooling2D(pool_size=(297, 1), padding='valid')(X_3_2)
    X_3_2 = Flatten()(X_3_2)
    X_3_3 = Conv2D(200, kernel_size=(3, 100), strides=(1, 1), activation='relu')(X_3)
    X_3_3 = layers.normalization.BatchNormalization(axis=-1)(X_3_3)
    X_3_3 = MaxPooling2D(pool_size=(296, 1), padding='valid')(X_3_3)
    X_3_3 = Flatten()(X_3_3)

    X_3 = layers.Concatenate(axis=-1)([X_3_1, X_3_2])
    X_3 = layers.Concatenate(axis=-1)([X_3, X_3_3])

    X = layers.Concatenate(axis=-1)([X_1, X_2])
    X = layers.Concatenate(axis=-1)([X, X_3])

    X = Dropout(0.6)(X)
    X = Dense(300, activation='relu')(X)
    X = layers.normalization.BatchNormalization(axis=-1)(X)

    X = Dropout(0.4)(X)
    X = Dense(100, activation='relu')(X)
    X = layers.normalization.BatchNormalization(axis=-1)(X)

    X = Dropout(0.4)(X)
    Y = Dense(1, activation='sigmoid')(X)
    model = Model(inputs=[q1, q2], outputs=Y, name='CNN_Model')

    print(model.summary())
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                       loss='binary_crossentropy', metrics=['accuracy'])

    return model

def predict(model,X_s1_char,X_s2_char):

    y1 = model.predict([X_s1_char,X_s2_char])
    y2 = model.predict([X_s1_char,X_s2_char])
    res = (y1 + y2)/2
    # print res[0:15]
    return res
