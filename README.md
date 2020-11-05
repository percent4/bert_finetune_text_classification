
# Text classification demo

利用Bert-finetune进行文本分类。  

## 数据集  

#### THUCNews数据集

使用THUCNews数据集进行训练与测试，10个分类，每个分类6500条数据。

类别如下：

体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐

数据集划分如下：

训练集: 5000 \* 10  
验证集: 500 \* 10  
测试集: 1000 \* 10

#### Sougou-Mini数据集

使用sougou_mini数据集进行训练与测试，5个分类，每个分类1000条数据。

类别如下：

体育, 健康, 汽车, 军事, 教育

数据集划分如下：

训练集: 800 \* 5
验证集: 100 \* 5  
测试集: 100 \* 5

#### 新闻标题分类数据集

使用约38w条新闻标题数据进行训练与测试，类别为15个，如下：

文化/娱乐/体育/财经/房产/汽车/教育/科技/军事/旅游/国际/证券/农业/游戏/民生

数据集划分如下：

训练集: 229612
验证集: 76538
测试集: 76538

### 模型参数及评估

模型：BERT-finetune

#### THUCNews数据集

模型参数：

```
max_seq_length = 512
learning_rate = 2e-5
train_batch_size = 12
num_train_epochs = 3
num_labels = 10
```

模型评估

```
Test loss: 0.114239, Test acc: 0.968100
              precision    recall  f1-score   support

          体育     1.0000    0.9970    0.9985      1000
          科技     0.9959    0.9800    0.9879      1000
          时政     0.9679    0.9350    0.9512      1000
          家居     0.9494    0.9010    0.9246      1000
          财经     0.9326    0.9690    0.9505      1000
          游戏     0.9513    0.9960    0.9731      1000
          时尚     0.9801    0.9370    0.9581      1000
          娱乐     0.9929    0.9760    0.9844      1000
          房产     0.9559    0.9960    0.9755      1000
          教育     0.9585    0.9940    0.9759      1000

    accuracy                         0.9681     10000
   macro avg     0.9685    0.9681    0.9680     10000
weighted avg     0.9685    0.9681    0.9680     10000
```


#### Sougou-Mini数据集

模型参数：

```
max_seq_length = 512
learning_rate = 2e-5
train_batch_size = 8
num_train_epochs = 5
num_labels = 5
```

模型评估

```
              precision    recall  f1-score   support

          汽车     0.9706    1.0000    0.9851        99
          健康     0.9800    0.9899    0.9849        99
          教育     1.0000    1.0000    1.0000        99
          军事     0.9796    0.9697    0.9746        99
          体育     1.0000    0.9697    0.9846        99

    accuracy                         0.9859       495
   macro avg     0.9860    0.9859    0.9858       495
weighted avg     0.9860    0.9859    0.9858       495
```

#### 新闻标题分类数据集

模型参数：

```
max_seq_length = 64
learning_rate = 2e-5
train_batch_size = 64
num_train_epochs = 3
num_labels = 15
```

模型评估

```
              precision    recall  f1-score   support

          民生     0.9441    0.9531    0.9486      7512
          财经     0.8854    0.8717    0.8785      5027
          农业     0.8856    0.8674    0.8764      3830
          游戏     0.8564    0.8323    0.8442      5409
          体育     0.9064    0.9115    0.9090      7925
          军事     0.9105    0.9294    0.9199      3569
          科技     0.9042    0.9056    0.9049      5380
          旅游     0.8783    0.8897    0.8839      5556
          房产     0.8478    0.8521    0.8499      4300
          汽车     0.8614    0.8248    0.8427      1273
          教育     0.9322    0.9344    0.9333      7106
          文化     0.9495    0.9046    0.9265      5841
          证券     0.8608    0.8921    0.8762      8314
          国际     0.3333    0.0441    0.0779        68
          娱乐     0.8092    0.8220    0.8156      5428

    accuracy                         0.8916     76538
   macro avg     0.8510    0.8290    0.8325     76538
weighted avg     0.8914    0.8916    0.8913     76538
```


### 模型服务

1. 将BERT中文预训练模型文件`chinese_L-12_H-768_A-12`放在model文件夹下；
2. 启动服务:

```
python3 bert_model_server.py
```

3. HTTP请求命令：

```
curl --location --request POST 'http://192.168.1.193:5000/cls' \
--header 'Content-Type: application/json' \
--data-raw '{
    "text": "《秦时明月世界》，原“腾讯秦时明月手游”，是由玄机授权，根据国漫代表作品—《秦时明月》系列动漫改编而成的3D MMORPG手机游戏，致力于呈现合纵连横、诸子百家争锋、华夏一统的大秦风貌，打造一个生动、可触碰的秦时世界。"
}'
```

输出结果：

```
{
  "code": 200,
  "message": "success",
  "data": {
    "text": "《秦时明月世界》，原“腾讯秦时明月手游”，是由玄机授权，根据国漫代表作品—《秦时明月》系列动漫改编而成的3D MMORPG手机游戏，致力于呈现合纵连横、诸子百家争锋、华夏一统的大秦风貌，打造一个生动、可触碰的秦时世界。",
    "label": "游戏"
  }
}
```
