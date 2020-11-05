# -*- coding: utf-8 -*-
# @Time : 2020/11/2 11:09
# @Author : Jclian91
# @File : bert_model_test.py
# @Place : Yangpu, Shanghai
import numpy as np
from sklearn.metrics import classification_report

from bert import modeling
from bert.data_loader import *
from util.cnews_loader import read_file

processors = {"cnews": CnewsProcessor}


# 将test写入至tf_record
def get_test_example(data_dir, train_file, dev_file, test_file):
    processor = processors[task_name](data_dir, train_file, dev_file, test_file)
    label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)

    examples = processor.get_test_examples()
    features = get_test_features(examples, label_list, max_seq_length, tokenizer)

    contents, true_labels = read_file(os.path.join(data_dir, test_file))

    return true_labels, features


def test_model(sess, graph, features):
    """
    :param sess:
    :param graph:
    :param features:
    :return:
    """

    total_loss = 0.0
    total_acc = 0.0

    input_ids = graph.get_operation_by_name('input_ids').outputs[0]
    input_mask = graph.get_operation_by_name('input_mask').outputs[0]
    segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
    labels = graph.get_operation_by_name('labels').outputs[0]
    is_training = graph.get_operation_by_name('is_training').outputs[0]
    loss = graph.get_operation_by_name('loss/loss').outputs[0]
    acc = graph.get_operation_by_name('accuracy/acc').outputs[0]

    predict_category_id = graph.get_operation_by_name('loss/pred').outputs[0]

    data_len = len(features)
    batch_size = 12
    num_batch = int((len(features) - 1) / batch_size) + 1

    predict_labels = []
    label_list = processors[task_name](data_dir, train_file, dev_file, test_file).get_labels()
    for i in range(num_batch):
        print(i)
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)

        batch_len = end_index-start_index

        _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
        _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
        _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
        _labels = np.array([data.label_id for data in features[start_index:end_index]])
        feed_dict = {input_ids: _input_ids,
                     input_mask: _input_mask,
                     segment_ids: _segment_ids,
                     labels: _labels,
                     is_training: False}
        test_loss, test_acc, pred_id = sess.run([loss, acc, predict_category_id], feed_dict=feed_dict)
        predict_labels.extend([label_list[_] for _ in pred_id])
        total_loss += test_loss * batch_len
        total_acc += test_acc * batch_len

    return predict_labels, total_loss / data_len, total_acc / data_len


# 模型评估
def test(graph_path, data_dir, train_file, dev_file, test_file):
    print("loading model...")
    true_labels, features = get_test_example(data_dir, train_file, dev_file, test_file)
    graph = tf.Graph()
    saver = tf.train.import_meta_graph("./model/bert/{}".format(graph_path), graph=graph)
    sess = tf.Session(graph=graph)
    saver.restore(sess, tf.train.latest_checkpoint("./model/bert"))
    predict_labels, test_loss, test_acc = test_model(sess, graph, features)
    print("Test loss: %f, Test acc: %f" % (test_loss, test_acc))
    print(classification_report(true_labels, predict_labels, target_names=list(set(true_labels)), digits=4))


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
    graph_path = "model-8000.meta"
    test(graph_path, data_dir, train_file, dev_file, test_file)