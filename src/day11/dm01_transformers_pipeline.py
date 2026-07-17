"""
案例:
    演示使用 Transformers库, 来实现NLP常见的6个任务.

关于 Transformers库, 主要有三种用法:
    1. 管道方式.
    2. 自动模型.
    3. 具体模型.

你要做的事儿:
    1. 共享资料中, 提供的预训练模型要先下载(同步), 然后解压(6个模型)
    2. 安装 Transformers 模块.
        pip install transformers
    3. 安装 datasets 模块.
        pip install datasets
"""

# 导包
import torch
from transformers import pipeline       # pipeline是Transformers库提供的高级接口, 可快速调用 预训练模型 完成常见的NLP任务.
import numpy as np


# todo 1.情感分类任务.
def dm01_test_classification():
    # 需求: 用 中文预训练模型进行 情感分析.

    # 1. 创建1个pipeline对象.
    # 参1: 任务类型, 这里是: 情感分析任务.
    # 参2: 指定模型的路径, 我用的是: 本地路径, 你可以用相对路径. 但是: 路径要合法(不能出现中文, 空格, 特殊符号等)
    my_model = pipeline(task='sentiment-analysis', model=r'C:\Software\CentOS_All\PretrainedModel\chinese_sentiment')
    # my_model = pipeline(task='sentiment-analysis', model='./model/chinese_sentiment')

    # 2. 将文本输入模型进行 情感分类.
    output = my_model('我爱北京天安门, 天安门上太阳升!')

    # 3. 打印结果.
    # 结果为: [{'label': 'star 5', 'score': 0.6365295648574829}]
    # 解释:           5星好评              分数(概率)
    print(f'output: {output}')


# todo 2.特征抽取任务 -> 属于: 不带任务头输出(模型把'通用特征'提取出来 -> 模型自己定义的 任务结果)
# 大白话解释:
#    不带任务头输出: 只给'原材料'(文本的原始特征), 适合: 你自己后续开发新任务.  例如: 买一堆手机零件, 你自己接着组装, 改装.
#    带头任务头输出: 直接给'成品'(针对具体任务的结果), 适合: 直接用模型完成指定任务. 例如: 买一部现成的能打电话, 刷视频的完整手机(直接用)
def dm02_test_feature_extraction():
    # 1. 创建1个pipeline对象.
    # 参1: 任务类型, 这里是: 特征提取(抽取)任务.
    # 参2: 指定模型的路径, 我用的是: 本地路径, 你可以用相对路径. 但是: 路径要合法(不能出现中文, 空格, 特殊符号等)
    my_model = pipeline(task='feature-extraction', model=r'C:\Software\CentOS_All\PretrainedModel\bert-base-chinese')

    # 2. 将文本输入模型进行 特征抽取.
    output = my_model('人生该如何起头')

    # 3. 打印结果.
    print(f'output: {output}')                    #
    print(f'类型: {type(output)}')                 # <class 'list'>
    print(f'形状: {np.array(output).shape}')       # [batch_size, seq_len, d_model] -> [1, 9, 768]

    # 思考: 不是7个字吗? 怎么是9个? -> 因为有 开头[CLS] 和 结尾[SEP], 只要用bert-base-chinese, 就会自动加1个开头和1个结束标记.



# todo 3. 完形填空 -> 一次只能预测1个[MASK], 如果要多个[MASK], 则必须通过循环实现.
# 例如: 我[MASK]你
# 例如: 我[MASK]你, 你[MASK]我    一次只能预测1个MASK, 如果要多个[MASK], 则必须通过循环实现.
def dm03_test_fill_mask():
    # 1. 创建1个pipeline对象.
    # 参1: 模型类型, 这里是: 完形填空任务.
    # 参2: 模型路径, 我用的是: 本地路径, 你可以用相对路径. 但是: 路径要合法(不能出现中文, 空格, 特殊符号等)
    my_model = pipeline(task='fill-mask', model=r'C:\Software\CentOS_All\PretrainedModel\chinese-bert-wwm')

    # 2. 将文本输入模型进行 完形填空.
    # output = my_model('我[MASK]你')
    output = my_model('我想明天去[MASK]家吃饭')


    # 3. 打印结果.
    print(f'output: {output}')

    # 结果如下
    """
    output: [
        {'score': 0.4332510530948639, 'token': 2695, 'token_str': '愛', 'sequence': '我 愛 你'}, 
        {'score': 0.19461670517921448, 'token': 3221, 'token_str': '是', 'sequence': '我 是 你'}, 
        {'score': 0.06783866137266159, 'token': 4263, 'token_str': '爱', 'sequence': '我 爱 你'}, 
        {'score': 0.02951931208372116, 'token': 1469, 'token_str': '和', 'sequence': '我 和 你'}, 
        {'score': 0.02699531987309456, 'token': 5645, 'token_str': '與', 'sequence': '我 與 你'}
    ]
    """


# todo 4.阅读理解. -> 抽取式问答, 输入一段文本和一个问题, 让模型输出结果.
def dm04_test_question_answering():
    # 1. 创建1个pipeline对象.
    # 参1: 任务类型, 这里是: 阅读理解任务.
    # 参2: 模型路径, 我用的是: 本地路径, 你可以用相对路径. 但是: 路径要合法(不能出现中文, 空格, 特殊符号等)
    my_model = pipeline(task='question-answering', model=r'C:\Software\CentOS_All\PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large')

    # 2. 给模型送数据, 做: 预测.
    # 2.1 定义问答句子.
    context = '我叫夯哥, 我是一名教师, 我的喜好是乒乓球和台球'
    questions = ['我叫啥?', '我是做什么的?', '我的爱好是什么?']

    # 2.2 将上述的数据传给模型.
    output = my_model(question=questions, context=context)

    # 3. 打印结果.
    print(f'output: \n{output}')

    # 结果格式如下:
    """
        [
            {'score': 0.9944425821304321, 'start': 2, 'end': 4, 'answer': '夯哥'}, 
            {'score': 0.9843989610671997, 'start': 10, 'end': 12, 'answer': '教师'}, 
            {'score': 0.07795657217502594, 'start': 19, 'end': 25, 'answer': '乒乓球和台球'}
        ]
    """


# todo 5. 文本摘要 -> '提炼精华'
def dm05_test_summarization():
    # 1. 创建1个pipeline对象.
    # 参1: 模型类型, 这里是: 文本摘要任务.
    # 参2: 模型路径, 我用的是: 本地路径, 你可以用相对路径. 但是: 路径要合法(不能出现中文, 空格, 特殊符号等)
    my_model = pipeline(task='summarization', model=r'C:\Software\CentOS_All\PretrainedModel\distilbart-cnn-12-6')

    # 2. 定义数据, 输入模型, 获取预测结果.
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    output = my_model(text)

    # 3. 输出结果.
    print(f'output: \n{output}')


# todo 6. NER任务 -> 命名实体识别, 在一段文本中找到特定的'实体'
# 字母解释: B -> 实体的开头, I -> 实体的内容/中间.  O: 不是实体(例如: 在, 的这些字)
# 命名实体识别: 全称是Named Entity Recognition, 也就是识别出这段文本中可能存在的<<命名实体>>,
# 例如: 人名(PER, person), 地名(LOC, location), 机构名(ORG, organization) 以及其它实体(MISC, miscellaneous)
def dm06_test_ner():
    # 1.创建1个pipeline对象.
    my_model = pipeline(task='ner', model=r'C:\Software\CentOS_All\PretrainedModel\roberta-base-finetuned-cluener2020-chinese')

    # 2.给模型发送数据, 获取预测结果.
    # output = my_model('我爱北京天安门, 天安门上太阳升!')
    output = my_model('我爱武汉黄鹤楼, 黄鹤楼上黄鹤飞!')

    # 3.输出结果.
    print(f'output: \n{output}')

    # 结果格式如下:
    """
        [
            {'entity': 'B-address', 'score': 0.89396185, 'index': 3, 'word': '北', 'start': 2, 'end': 3}, 
            {'entity': 'I-address', 'score': 0.87884074, 'index': 4, 'word': '京', 'start': 3, 'end': 4}, 
            {'entity': 'I-address', 'score': 0.5274998, 'index': 5, 'word': '天', 'start': 4, 'end': 5}, 
            {'entity': 'I-address', 'score': 0.7829927, 'index': 6, 'word': '安', 'start': 5, 'end': 6}, 
            {'entity': 'I-address', 'score': 0.7299224, 'index': 7, 'word': '门', 'start': 6, 'end': 7}, 
            
            {'entity': 'B-address', 'score': 0.6551384, 'index': 9, 'word': '天', 'start': 9, 'end': 10}, 
            {'entity': 'I-address', 'score': 0.53630906, 'index': 10, 'word': '安', 'start': 10, 'end': 11}, 
            {'entity': 'I-address', 'score': 0.51451707, 'index': 11, 'word': '门', 'start': 11, 'end': 12}
        ]
    """



# todo n.测试代码.
if __name__ == '__main__':
    # 1. 测试: 情感分类.
    # dm01_test_classification()

    # 2. 测试: 特征抽取.
    # dm02_test_feature_extraction()

    # 3. 测试: 完形填空.
    # dm03_test_fill_mask()

    # 4. 测试: 阅读理解.
    # dm04_test_question_answering()

    # 5. 测试: 文本摘要.
    # dm05_test_summarization()

    # 6. 测试: NER任务.
    dm06_test_ner()