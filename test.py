# -*- coding: utf-8 -*-
# @Time : 2020/11/2 10:23
# @Author : Jclian91
# @File : test.py
# @Place : Yangpu, Shanghai
import pandas as pd

# 将txt处理成csv格式
# labels = []
# contents = []
# with open("./data/cnews/cnews.test.txt", "r", encoding="utf-8") as f:
#     lines = f.readlines()
#
# for line in lines:
#     labels.append(line.split("\t")[0])
#     contents.append(line.split("\t")[-1])
#
# df = pd.DataFrame({"label": labels, "content": contents})
# df.to_csv("cnews_test.csv", index=False)


csv_file_path = "./data/cnews/cnews_test.csv"
df = pd.read_csv(csv_file_path)
print(df.label.unique())

