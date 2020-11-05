# -*- coding: utf-8 -*-
# @Time : 2020/11/3 18:37
# @Author : Jclian91
# @File : bert_model_server.py
# @Place : Yangpu, Shanghai
import json
import tensorflow as tf
import numpy as np
from flask import Flask
from flask import request
import traceback

from bert.data_loader import InputExample, get_test_features, CnewsProcessor
from bert import tokenization

# 加载模型
graph_path = "model-2000.meta"
graph = tf.Graph()
saver = tf.train.import_meta_graph("./model/bert/{}".format(graph_path), graph=graph)
sess = tf.Session(graph=graph)
saver.restore(sess, tf.train.latest_checkpoint("./model/bert"))

input_ids = graph.get_operation_by_name('input_ids').outputs[0]
input_mask = graph.get_operation_by_name('input_mask').outputs[0]
segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
labels = graph.get_operation_by_name('labels').outputs[0]
is_training = graph.get_operation_by_name('is_training').outputs[0]
predict_category_id = graph.get_operation_by_name('loss/pred').outputs[0]

# 将文本转化为向量特征
with open("labels.json", "r", encoding="utf-8") as f:
    label_list = json.loads(f.read())

processors = {"cnews": CnewsProcessor}
tokenizer = tokenization.FullTokenizer(vocab_file="./model/chinese_L-12_H-768_A-12/vocab.txt")


app = Flask(__name__)


# 封装成HTTP服务
@app.route("/cls", methods=['GET',  'POST'])
def model_predict():
    return_result = {"code": 200, "message": "success", "data": {}}
    try:
        if request.method == 'GET':
            text = request.args.get("text")
        elif request.method == 'POST':
            text = request.get_json()["text"]
        else:
            raise Exception("Missing argument text!")

        text_a = tokenization.convert_to_unicode(text)
        label = tokenization.convert_to_unicode(label_list[0])
        examples = [InputExample(guid="pred-0", text_a=text_a, text_b=None, label=label)]
        features = get_test_features(examples, label_list, 512, tokenizer)

        data_len = len(features)
        batch_size = 12
        num_batch = int((len(features) - 1) / batch_size) + 1

        # 文本分类预测
        for i in range(num_batch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_len)
            batch_len = end_index - start_index

            _input_ids = np.array([data.input_ids for data in features[start_index:end_index]])
            _input_mask = np.array([data.input_mask for data in features[start_index:end_index]])
            _segment_ids = np.array([data.segment_ids for data in features[start_index:end_index]])
            _labels = np.array([data.label_id for data in features[start_index:end_index]])
            feed_dict = {input_ids: _input_ids,
                         input_mask: _input_mask,
                         segment_ids: _segment_ids,
                         labels: _labels,
                         is_training: False}
            predict_id_array = sess.run([predict_category_id], feed_dict=feed_dict)
            predict_id = predict_id_array[0].tolist()[0]
            return_result["data"] = {"text": text, "label": label_list[predict_id]}

    except Exception:
        return_result["code"] = 400
        return_result["message"] = traceback.format_exc()

    return json.dumps(return_result, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
