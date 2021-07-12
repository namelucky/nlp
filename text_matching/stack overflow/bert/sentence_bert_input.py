import logging

logging.basicConfig(level=logging.ERROR)

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import *
import pickle

java_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('java'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
c_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('c++'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
html_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('html'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
objc_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('objective-c'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
python_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('python'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")
ruby_df = pd.read_csv("../divide/2018/mypair_{}_train.csv".format('ruby'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n")

df_train=java_df.append(c_df)
df_train.append(c_df)
df_train.append(html_df)
df_train.append(objc_df)
df_train.append(python_df)
df_train.append(ruby_df)

java_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('java'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)
c_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('c++'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)
html_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('html'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)
objc_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('objective-c'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)
python_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('python'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)
ruby_test_df = pd.read_csv("../divide/2018/mypair_{}_test.csv".format('ruby'), sep='\t', names=["id", "question1", "question2", "duplicated", "s1_title", "s2_title", "s1_body", "s2_body", "maintag"], encoding='utf-8-sig',lineterminator="\n",header=0)

df_test=java_test_df.append(c_test_df)
df_test.append(c_test_df)
df_test.append(html_test_df)
df_test.append(objc_test_df)
df_test.append(python_test_df)
df_test.append(ruby_test_df)

print(df_test.isnull().any())  # 检查是否有缺失值
print(df_train.isnull().any())

PATH = r'/home/znuser2/lwl/pretrained_model/en/bert_h5/'
BERT_PATH = PATH
WEIGHT_PATH = PATH
MAX_SEQUENCE_LENGTH = 20
input_categories = ['s1_title', 's2_title']
output_categories = 'duplicated'


def _convert_to_transformer_inputs(mystr,  tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(mystr, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str(mystr),
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       # truncation=True
                                       )

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_id(
        mystr, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []

    for index, row in tqdm(df.iterrows()):
        mystr= df.loc[index,columns]

        ids_q, masks_q, segments_q = \
            _convert_to_transformer_inputs(mystr, tokenizer, max_sequence_length)

        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)

    return [np.asarray(input_ids_q, dtype=np.int32),
            np.asarray(input_masks_q, dtype=np.int32),
            np.asarray(input_segments_q, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


tokenizer = BertTokenizer.from_pretrained(BERT_PATH + 'vocab.txt')
outputs = compute_output_arrays(df_train, output_categories)

inputs_q = compute_input_arrays(df_train, 's1_title', tokenizer, MAX_SEQUENCE_LENGTH)
inputs_a = compute_input_arrays(df_train, 's2_title', tokenizer, MAX_SEQUENCE_LENGTH)
print(outputs.shape)



test_inputs_q = compute_input_arrays(df_test, 's1_title', tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs_a = compute_input_arrays(df_test, 's2_title', tokenizer, MAX_SEQUENCE_LENGTH)
test_outputs=compute_output_arrays(df_test, output_categories)

def save(nfile, object):
    with open(nfile, 'wb') as file:
        pickle.dump(object, file)


test_label=df_test['duplicated']
save('./bert_input/test_label.pickle',test_label)



np.save('./bert_input/inputs_q.npy', inputs_q)
np.save('./bert_input/inputs_a.npy', inputs_a)
np.save('./bert_input/outputs.npy', outputs)

np.save('./bert_input/test_inputs_q.npy', test_inputs_q)
np.save('./bert_input/test_inputs_a.npy', test_inputs_a)
np.save('./bert_input/test_outputs.npy', test_outputs)

