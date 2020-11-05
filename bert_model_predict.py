# -*- coding: utf-8 -*-
# @Time : 2020/10/29 22:45
# @Author : Jclian91
# @File : bert_model_predict.py
# @Place : Yangpu, Shanghai
import json
import tensorflow as tf
import numpy as np

from bert.data_loader import InputExample, get_test_features, CnewsProcessor
from bert import tokenization

# 加载模型
graph_path = "model-7000.meta"
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

texts = ["近日，被誉为央视第一美女主持的刘芳菲在自己的社交平台上跟粉丝们分享了一组自己的最新的诗意自拍照，照片中，刘芳菲站在桥上看着远处的夕阳，画面非常的养眼，上身穿着一件白色衬衫打底，最上面两颗纽扣微微敞开，随意中又增添了几分小性感，外面则是一件绿色的西装，双排的金色纽扣为造型增添了几分金属元素，给人眼前一亮的感觉。43岁刘芳菲身材真极品！穿黑裙戴翡翠贵气挡不住，生活诗意超羡慕",
         "9月16日， 2021春夏纽约时装周落下帷幕。受当地疫情影响，这场时装周被压缩到了四天，并且，Ralph Lauren、Tommy Hilfiger、Michael Kors、Marc Jacobs、Oscar de la Renta、Tory Burch等知名品牌都缺席了本届时装周。其他参与的品牌也都选择在 CFDA（美国时装设计师协会）自有平台“Runway 360”和Instagram上通过直播的形式展示新系列。",
         "11月3日，2020-21赛季CBA联赛常规赛第8轮拉开战幕，四川男篮对阵福建男篮。四川队90-80战胜福建队从而夺取近5轮首胜。福建队输掉此役后遭遇开局7连败，继续刷新队史最差开局纪录。得分方面，哈达迪18分17板7助，苏若禹16分7板4助，袁堂文13分4板4助4断，景菡一22分10板3断，陈辰11分2助；王哲林17分8板5助，胡珑贸18分5板4助4断，黄毅超14分6板3助，陈林坚5分2助，于长春12分5板，刘洺宇10分2板。",
         "北京时间11月4日凌晨4点，2020-2021赛季欧洲足球冠军联赛小组赛第3轮打响一场焦点战役，西甲豪门皇家马德里坐镇主场迎战意甲劲旅国际米兰。齐达内已经公布了参加本场比赛的球员名单，卡瓦哈尔、奥德里奥佐拉、纳乔、米利唐合计4位后卫因为身体原因无法出战。国米方面则是损失了锋线悍将卢卡库。",
         "“不辅导作业母慈子孝，一辅导作业鸡飞狗跳。”相信每位爸爸妈妈应该都亲身经历过，而且个中滋味汇总结起来不言而喻。然而，在孩子眼中，你的“陪读”是什么样的？近日，鄞州区实验小学施燕婷老师在学校里做了一个小调查， 发现 大多数孩子表示不愿意让父母在旁边陪自己写作业。",
         "九游研究所是九游APP大神玩家自发组织的团队，专注热门资讯跟踪、游戏玩法研究，以及个人攻略心得分享，希望每个玩家都能玩得更开心、更省心！今天继续给大家来扫盲千秋辞的基础攻略。前两天梳理了【控制英雄】和【回血英雄】，那么接下来说一说，千秋辞里都有哪些增益和减益辅助功能的英雄呢？辅助英雄无论是之前的放置奇兵也好，现在千秋辞也罢，在PVE当中其实表现得非常明显。而PVP的当中回血英雄和控制英雄会显得更重要一些。因为这两块内容有讲过，所以会稍微略带过即可。",
         "据中国证监会消息，11月2日，中国人民银行、中国银保监会、中国证监会、国家外汇管理局对蚂蚁集团实际控制人马云、董事长井贤栋、总裁胡晓明进行了监管约谈。对此，蚂蚁集团回应表示：蚂蚁集团实际控制人与相关管理层接受了各主要监管部门的监管约谈。蚂蚁集团会深入落实约谈意见，继续沿着“稳妥创新、拥抱监管、服务实体、开放共赢”的十六字指导方针，继续提升普惠服务能力，助力经济和民生发展。",
         "今年上海新房市场火热程度肉眼可见，而且，千人摇、日光盘经常扎堆出现。热销项目背后大多都有不少共性，外在因素主要在于板块本身现状较为宜居乐业，而且有重磅规划赋能，升值潜力可期；内在因素大多是项目产品力突出，适合家庭改善置业。地处旗忠别墅区，15年来首个新中式项目【玖玺】就称得上是一个现象级爆款。6月27日首开，次月就登顶闵行区新房成交面积、成交套数、成交金额销冠，而且遥遥领先第二名（数据来自克尔瑞）。",
         "11月2日，习近平总书记主持召开中央全面深化改革委员会第十六次会议。会上，总书记作出了上述重要判断，并强调贯彻党的十九届五中全会精神，要有三个“深刻认识”，为“推动改革和发展深度融合、高效联动”指明前进方向。",
         "第三届中国国际进口博览会即将在上海举行。本届进博会电力保障工作中应用了5G、AR、人工智能、北斗导航、物联网等技术，提升保障水平。今年的进博会电力保障更多体现了数字新基建与电网技术的深度融合，在距离国家会展中心不到1公里的一处变电站，工作人员的装备融合了5G＋AR技术，通过佩戴的AR眼镜，实时查看和采集设备运行工况，并进行信息上报。这是一辆融合了5G通信、北斗卫星导航和电力物联网等技术的应急指挥车，它将直接应用在进博会主场馆——国家会展中心——的供电保障中。在这辆指挥车上，工作人员可以实时了解到国家会展中心乃至整个上海电网的各类主要供电设备的运行状况，通过5G单兵系统实时下达指令，为现场工作人员维护抢修提供支持。本届进博会在总结前两届进博会电力保障工作经验的基础上，国网上海市电力公司迭代升级了全景智慧保电3.0系统。整个电网供电路径、实时负荷、异常报警及抢修进度等信息均以动态方式在交互大屏呈现。目前，分布在全市的129个抢修基地、173个抢修驻点、5000余名抢修人员、近1000台应急抢修车辆也已全部到位，以最佳状态迎接进博会的到来。"]

examples = []
for i, text in enumerate(texts):
    text_a = tokenization.convert_to_unicode(text)
    label = tokenization.convert_to_unicode(label_list[0])
    examples.append(InputExample(guid="pred-%s" % i, text_a=text_a, text_b=None, label=label))


features = get_test_features(examples, label_list, 512, tokenizer)

data_len = len(features)
batch_size = 12
num_batch = int((len(features) - 1) / batch_size) + 1

# 文本分类预测
for i in range(num_batch):
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
    predict_id_array = sess.run([predict_category_id], feed_dict=feed_dict)
    # print(predict_id)
    for j, predict_id in enumerate(predict_id_array[0].tolist()):
        print("样本: %s, 分类结果: %s" % (j+1, label_list[predict_id]))

