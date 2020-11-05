# -*- coding: utf-8 -*-
import numpy as np
from datetime import datetime

from bert import modeling
from bert.data_loader import *


processors = {"cnews": CnewsProcessor}
tf.logging.set_verbosity(tf.logging.INFO)


class BertModel:
    def __init__(self, bert_config, num_labels, seq_length, init_checkpoint):
        self.bert_config = bert_config
        self.num_labels = num_labels
        self.seq_length = seq_length

        self.input_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='input_ids')
        self.input_mask = tf.placeholder(tf.int32, [None, self.seq_length], name='input_mask')
        self.segment_ids = tf.placeholder(tf.int32, [None, self.seq_length], name='segment_ids')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate = tf.placeholder(tf.float32, name='learn_rate')

        self.model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        self.inference()

    def inference(self):

        output_layer = self.model.get_pooled_output()

        with tf.variable_scope("loss"):
            def apply_dropout_last_layer(output_layer):
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
                return output_layer

            def not_apply_dropout(output_layer):
                return output_layer

            output_layer = tf.cond(self.is_training, lambda: apply_dropout_last_layer(output_layer),
                                   lambda: not_apply_dropout(output_layer))
            self.logits = tf.layers.dense(output_layer, self.num_labels, name='fc')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1, name="pred")

            one_hot_labels = tf.one_hot(self.labels, depth=self.num_labels, dtype=tf.float32)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=one_hot_labels)
            self.loss = tf.reduce_mean(cross_entropy, name="loss")
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # 准确率
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="acc")


# 将训练数据和测试数据写入tf_record中
def make_tf_record(output_dir, data_dir, train_file, dev_file, test_file, vocab_file):
    tf.gfile.MakeDirs(output_dir)
    processor = processors[task_name](data_dir, train_file, dev_file, test_file)
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
    train_file = os.path.join(output_dir, "train.tf_record")
    eval_file = os.path.join(output_dir, "eval.tf_record")

    # save data to tf_record
    train_examples = processor.get_train_examples()
    file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)

    # eval data
    eval_examples = processor.get_dev_examples()
    file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file)

    del train_examples, eval_examples


# Decodes a record to a TensorFlow example.
def _decode_record(record, name_to_features):
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read_data(data, batch_size, is_training, num_epochs):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.

    if is_training:
        data = data.shuffle(buffer_size=50000)
        data = data.repeat(num_epochs)

    data = data.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))
    return data


# 评估 val data 的准确率和损失
def evaluate(sess, model):
    # dev data
    test_record = tf.data.TFRecordDataset("./model/bert/eval.tf_record")
    test_data = read_data(test_record, train_batch_size, False, 3)
    test_iterator = test_data.make_one_shot_iterator()
    test_batch = test_iterator.get_next()

    data_nums = 0
    total_loss = 0.0
    total_acc = 0.0
    while True:
        try:
            features = sess.run(test_batch)
            feed_dict = {model.input_ids: features["input_ids"],
                         model.input_mask: features["input_mask"],
                         model.segment_ids: features["segment_ids"],
                         model.labels: features["label_ids"],
                         model.is_training: False,
                         model.learning_rate: learning_rate}

            batch_len = len(features["input_ids"])
            data_nums += batch_len

            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        except Exception as e:
            print(e)
            break

    return total_loss / data_nums, total_acc / data_nums


def train(train_batch_size, num_epochs):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    with tf.Graph().as_default():
        # train data
        train_record = tf.data.TFRecordDataset("./model/bert/train.tf_record")
        train_data = read_data(train_record, train_batch_size, True, num_epochs)
        train_iterator = train_data.make_one_shot_iterator()

        model = BertModel(bert_config, num_labels, max_seq_length, init_checkpoint)
        sess = tf.Session()
        saver = tf.train.Saver()
        train_steps = 0
        val_loss = 0.0
        val_acc = 0.0
        best_acc_val = 0.0
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            train_batch = train_iterator.get_next()
            while True:
                try:
                    train_steps += 1
                    features = sess.run(train_batch)
                    feed_dict = {model.input_ids: features["input_ids"],
                                 model.input_mask: features["input_mask"],
                                 model.segment_ids: features["segment_ids"],
                                 model.labels: features["label_ids"],
                                 model.is_training: True,
                                 model.learning_rate: learning_rate}
                    _, train_loss, train_acc = sess.run([model.optim, model.loss, model.acc],
                                                        feed_dict=feed_dict)

                    if train_steps % 1000 == 0:
                        val_loss, val_acc = evaluate(sess, model)

                    if val_acc > best_acc_val:
                        # 保存最好结果
                        best_acc_val = val_acc
                        saver.save(sess, "./model/bert/model", global_step=train_steps)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    now_time = datetime.now()
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train Acc: {2:>7.4%},' \
                          + ' Val Loss: {3:>6.4}, Val Acc: {4:>7.4%}, Time: {5} {6}'
                    print(msg.format(train_steps, train_loss, train_acc, val_loss, val_acc, now_time, improved_str))
                except Exception as e:
                    print(e)
                    break


if __name__ == "__main__":
    # 模型配置文件
    data_dir = "data/title_news"
    train_file = "title_train.csv"
    dev_file = "title_eval.csv"
    test_file = "title_test.csv"
    output_dir = "model/bert"
    task_name = "cnews"

    # 参数配置
    vocab_file = "./model/chinese_L-12_H-768_A-12/vocab.txt"
    bert_config_file = "./model/chinese_L-12_H-768_A-12/bert_config.json"
    init_checkpoint = "./model/chinese_L-12_H-768_A-12/bert_model.ckpt"
    max_seq_length = 64
    learning_rate = 2e-5
    train_batch_size = 64
    num_train_epochs = 3
    num_labels = 15

    # 模型训练
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    make_tf_record(output_dir, data_dir, train_file, dev_file, test_file, vocab_file)
    train(train_batch_size, num_epochs=num_train_epochs)

