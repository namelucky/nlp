# -*-coding=utf-8-*-
import json
import numpy as np
from tqdm import tqdm
from transformers import *
from transformers import RobertaTokenizer, TFRobertaModel

import logging
from loadlog import configure_logging
configure_logging("logging_config.json")

# MODEL_PATH = r'/root/lwl/pretrained-model/roberta/'
MODEL_PATH =  r'nghuyong/ernie-1.0'
base_path = r'/root/lwl/souhu/datasets'


def load(s):
    '''
    将raw_data的每一行数据的source,traget,label分别提取出来
    :param s:
    :return:
    '''
    tmp = json.loads(s)
    if "source" not in tmp or "target" not in tmp or ("labelA" not in tmp and "labelB" not in tmp):
        raise ValueError("sample is incomplete")
    if "labelA" in tmp:
        label = int(tmp["labelA"])
    else:
        label = int(tmp["labelB"])
    source = tmp["source"]
    target = tmp["target"]
    return source, target, label


def load_train_data(train_path, valid_path):
    '''
    将训练集和验证集全部的source,traget,label分别提出为单独的list
    :param train_path:
    :param valid_path:
    :return:
    '''
    source = []
    traget = []
    lable = []

    with open(train_path, encoding='utf-8') as f:
        for line in f:
            s, t, l = load(line)
            source.append(s)
            traget.append(t)
            lable.append(l)

    with open(valid_path, encoding='utf-8') as f:
        for line in f:
            s, t, l = load(line)
            source.append(s)
            traget.append(t)
            lable.append(l)

    return source, traget, lable


def load_test_data(test_path):
    '''
    将测试集全部的source,traget,id分别提出为单独的list
    :param test_path:
    :return:
    '''
    source = []
    target = []
    id = []

    with open(test_path, encoding='utf-8') as f:
        for line in f:
            tmp = json.loads(line)
            source.append(tmp['source'])
            target.append(tmp['target'])
            id.append(tmp['id'])

    return source, target, id


def convert_to_bert_inputs(source, target, tokenizer, max_sequence_length):
    '''
    将单条source,target匹配句子转换为bert的输入形式
    :param source:
    :param target:
    :param tokenizer:
    :param max_sequence_length:
    :return:
    '''

    def return_input(str1, str2, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1, str2,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy,
                                       truncation=True,
                                       )
        # 根据Transformers自带分词获取需要的输入信息
        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]

        # 根据设定的Max_len获取padding的长度
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id

        # 将不足max_len的输入部分进行填充
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids_q, input_masks_q, input_segments_q = return_input(
        source, target, 'longest_first', max_sequence_length)

    return [input_ids_q, input_masks_q, input_segments_q]


def get_input(source, target, tokenizer, source_length, target_length):
    '''
    将全部的source和target转换为bert处理的输入形式
    :param source:
    :param target:
    :param tokenizer:
    :param max_sequence_length:
    :return:
    '''
    input_ids, input_masks, input_segments = [], [], []

    for s, t in tqdm(zip(source, target)):
        ids, masks, segments = convert_to_bert_inputs(s[:source_length], t[:target_length], tokenizer,
                                                      source_length + target_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)]


def process_data():
    source_length_list = [256, 256, 256]
    target_lengh_list = [256, 256, 256]

    A_train_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配A类/train.txt',
                         base_path + r'/sohu2021_open_data_clean/短长匹配A类/train.txt',
                         base_path + r'/sohu2021_open_data_clean/长长匹配A类/train.txt',
                         base_path + r'/round2/长长匹配A类.txt',
                         base_path + r'/round2/短长匹配A类.txt',
                         base_path + r'/round2/短短匹配A类.txt',
                         base_path + r'/round3/长长匹配A类/train.txt',
                         base_path + r'/round3/短长匹配A类/train.txt',
                         base_path + r'/round3/短短匹配A类/train.txt',

                         ]

    A_vaild_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配A类/valid.txt',
                         base_path + r'/sohu2021_open_data_clean/短长匹配A类/valid.txt',
                         base_path + r'/sohu2021_open_data_clean/长长匹配A类/valid.txt', ]

    A_test_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配A类/test_with_id.txt',
                        base_path + r'/sohu2021_open_data_clean/短长匹配A类/test_with_id.txt',
                        base_path + r'/sohu2021_open_data_clean/长长匹配A类/test_with_id.txt', ]

    A_name_list = ['short_short_A',
                   'short_long_A',
                   'long_long_A', ]

    A_train_source, A_train_target, A_train_lable, A_test_source, A_test_target, A_test_id = [], [], [], [], [], []

    for train_path, valid_path, test_path, name, source_length, target_lengh in zip(A_train_path_list,
                                                                                    A_vaild_path_list,
                                                                                    A_test_path_list, A_name_list,
                                                                                    source_length_list,
                                                                                    target_lengh_list):
        # 加载原始训练集和测试集
        train_source, train_target, train_lable = load_train_data(train_path, valid_path)
        test_source, test_target, test_id = load_test_data(test_path)

        A_train_source.extend(train_source)
        A_train_target.extend(train_target)
        A_train_lable.extend(train_lable)
        A_test_source.extend(test_source)
        A_test_target.extend(test_target)
        A_test_id.extend(test_id)

    # 原始数据数字化
    tokenizer =  BertTokenizer.from_pretrained(MODEL_PATH)


    A_train_input = get_input(A_train_source, A_train_target, tokenizer, 256, 256)
    logging.info("A_train_input")
    logging.info(A_train_input)
    A_train_lable = np.asarray(A_train_lable)
    logging.info("A_train_lable")
    logging.info(A_train_lable)
    A_test_input = get_input(A_test_source, A_test_target, tokenizer, 256, 256)
    A_test_id = np.asarray(A_test_id)

    # 存储数据
    np.save('data/A_train_input.npy', A_train_input)
    np.save('data/A_train_label.npy', A_train_lable)
    np.save('data/A_test_input.npy', A_test_input)
    np.save('data/A_test_id.npy', A_test_id)

    # -------------------------------------------------

    B_train_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配B类/train.txt',
                         base_path + r'/sohu2021_open_data_clean/短长匹配B类/train.txt',
                         base_path + r'/sohu2021_open_data_clean/长长匹配B类/train.txt',
                         base_path + r'/round2/长长匹配B类.txt',
                         base_path + r'/round2/短长匹配B类.txt',
                         base_path + r'/round2/短短匹配B类.txt',
                         base_path + r'/round3/长长匹配B类/train.txt',
                         base_path + r'/round3/短长匹配B类/train.txt',
                         base_path + r'/round3/短短匹配B类/train.txt',
                         ]

    B_vaild_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配B类/valid.txt',
                         base_path + r'/sohu2021_open_data_clean/短长匹配B类/valid.txt',
                         base_path + r'/sohu2021_open_data_clean/长长匹配B类/valid.txt', ]

    B_test_path_list = [base_path + r'/sohu2021_open_data_clean/短短匹配B类/test_with_id.txt',
                        base_path + r'/sohu2021_open_data_clean/短长匹配B类/test_with_id.txt',
                        base_path + r'/sohu2021_open_data_clean/长长匹配B类/test_with_id.txt', ]

    B_name_list = ['short_short_B',
                   'short_long_B',
                   'long_long_B', ]

    B_train_source, B_train_target, B_train_lable, B_test_source, B_test_target, B_test_id = [], [], [], [], [], []

    for train_path, valid_path, test_path, name, source_length, target_lengh in zip(B_train_path_list,
                                                                                    B_vaild_path_list,
                                                                                    B_test_path_list, B_name_list,
                                                                                    source_length_list,
                                                                                    target_lengh_list):
        # 加载原始训练集和测试集
        train_source, train_target, train_lable = load_train_data(train_path, valid_path)
        test_source, test_target, test_id = load_test_data(test_path)

        B_train_source.extend(train_source)
        B_train_target.extend(train_target)
        B_train_lable.extend(train_lable)
        B_test_source.extend(test_source)
        B_test_target.extend(test_target)
        B_test_id.extend(test_id)

    # 原始数据数字化
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH )

    B_train_input = get_input(B_train_source, B_train_target, tokenizer, 256, 256)
    B_train_lable = np.asarray(B_train_lable)

    B_test_input = get_input(B_test_source, B_test_target, tokenizer, 256, 256)
    B_test_id = np.asarray(B_test_id)

    # 存储数据
    np.save('data/B_train_input.npy', B_train_input)
    np.save('data/B_train_label.npy', B_train_lable)
    np.save('data/B_test_input.npy', B_test_input)
    np.save('data/B_test_id.npy', B_test_id)

    # 合并A,B数据集同时增加一个type
    A_train_type = [[0]] * len(A_train_lable)
    B_train_type = [[1]] * len(B_train_lable)

    A_test_type = [[0]] * len(A_test_id)
    B_test_type = [[1]] * len(B_test_id)

    all_train_input = [np.concatenate((A_train_input[0], B_train_input[0])),
                       np.concatenate((A_train_input[1], B_train_input[1])),
                       np.concatenate((A_train_input[2], B_train_input[2])),
                       np.concatenate((A_train_type, B_train_type))]
    all_train_label = np.concatenate((A_train_lable, B_train_lable))

    #     for i in all_train_input:
    #         print(i.shape)

    all_test_input = [np.concatenate((A_test_input[0], B_test_input[0])),
                      np.concatenate((A_test_input[1], B_test_input[1])),
                      np.concatenate((A_test_input[2], B_test_input[2])),
                      np.concatenate((A_test_type, B_test_type))]
    all_test_id = [np.concatenate((A_test_id, B_test_id))]

    # 存储数据
    np.save('data/all_train_input.npy', all_train_input[:3])
    np.save('data/all_train_input_type.npy', all_train_input[3])
    np.save('data/all_train_label.npy', all_train_label)

    np.save('data/all_test_input.npy', all_test_input[:3])
    np.save('data/all_test_input_type.npy', all_test_input[3])
    logging.info("all_test_id")
    logging.info(all_test_id)
    np.save('data/all_test_id.npy', all_test_id)


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


if __name__ == '__main__':
    process_data()
    # load_data()
