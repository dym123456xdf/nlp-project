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





# todo n.测试代码.
if __name__ == '__main__':
    # 1. 测试: 情感分类.
    # dm01_test_classification()

    # 2. 测试: 特征抽取.
    # dm02_test_feature_extraction()

    # 3. 测试: 完形填空.
    dm03_test_fill_mask()