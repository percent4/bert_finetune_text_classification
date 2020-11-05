# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import collections


# 读取数据文件
def read_file(file_name):
    df = pd.read_csv(file_name).fillna(value="")
    return df.content, df.label


# 根据训练数据构建词汇表并存为txt文件
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """
    :param train_dir: 训练数据路径
    :param vocab_dir: 词汇表存储路径
    :param vocab_size: 词汇表大小
    :return:
    """
    data_train, _ = read_file(train_dir)

    all_data = []
    # 将字符串转为单个字符的list
    for content in data_train:
        for word in content:
            if word.strip():
                all_data.append(word)

    counter = collections.Counter(all_data)
    counter_pairs = counter.most_common(vocab_size - 2)
    words, _ = list(zip(*counter_pairs))
    words = ['<UNK>'] + list(words)
    words = ['<PAD>'] + list(words)

    with open(vocab_dir, "a") as f:
        f.write('\n'.join(words) + "\n")

    return 0


def word_2_id(vocab_dir):
    """
    :param vocab_dir:
    :return:
    """
    with open(vocab_dir) as f:
        words = [_.strip() for _ in f.readlines()]

    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word


# categeory to id dict and id to category dict
def cat_2_id(categories):
    cat_to_id = dict(zip(categories, range(len(categories))))
    id_to_cat = dict((v, k) for k, v in cat_to_id.items())
    return cat_to_id, id_to_cat


def process_file(data_dir, word_to_id, cat_to_id, seq_length=512):
    """
    :param data_dir:
    :param word_to_id:
    :param cat_to_id:
    :param seq_length:
    :return:
    """
    contents, labels = read_file(data_dir)

    data_id, label_id = [], []
    for i in range(len(contents)):
        sent_ids = [word_to_id.get(w) if w in word_to_id else word_to_id.get("<UNK>") for w in contents[i]]
        # pad to the required length
        if len(sent_ids) > seq_length:
            sent_ids = sent_ids[:seq_length]
        else:
            padding = [0] * (seq_length - len(sent_ids))
            sent_ids += padding
        data_id.append(sent_ids)
        y_pad = [0] * len(cat_to_id)
        y_pad[cat_to_id[labels[i]]] = 1
        label_id.append(y_pad)

    return np.array(data_id), np.array(label_id)


def batch_iter(x, y, batch_size=32, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[shuffle_indices]
        y_shuffle = y[shuffle_indices]
    else:
        x_shuffle = x
        y_shuffle = y
    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])

# file_path = "../data/sougou_mini/sougou_test.csv"
# c, l = read_file(file_path)
# print(c)
# print(l)