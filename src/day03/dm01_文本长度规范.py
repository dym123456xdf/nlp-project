"""
案例:
    演示文本长度规范案例.

文本长度规范解释:
    概述:
        一般模型的输入需要 等尺寸大小的矩阵, 所以需要对 超长文本做截断, 对不足文本进行补齐.
    实现方式:
        方式1: 第三方包
            tensorflow#sequence
        方式2: 纯Python基础代码实现
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导包
from tensorflow.keras.preprocessing import sequence


# 定义遍历, 记录: 截断补齐长度参数.
cutlen = 10         # 实际开发中, 根据你的语料库的句子长度分布来自定义.

# todo 1.定义函数, 对输入的文本张量进行截断补齐. -> 第三方包
def padding(x_train):
    # 参1: 待处理的文本张量.
    # 参2: 最大长度.
    # 参3: 截断策略, pre(默认, 从序列前端截取), post(从序列后端截取)
    # 参4: 填充策略,  pre(默认, 从序列前端补齐)   post(从序列后端补齐)
    # return sequence.pad_sequences(x_train, maxlen=cutlen)
    # return sequence.pad_sequences(x_train, maxlen=cutlen, truncating='pre', padding='pre')
    return sequence.pad_sequences(x_train, maxlen=cutlen, truncating='post', padding='post')


# todo 2. 定义函数, 对输入的文本张量进行截断补齐. -> 纯Python基础代码实现
def padding_custom(x_train):
    # 1. 定义遍历, 记录: 初始化列表.
    list1 = []
    # 2. 遍历语料库, 获取到每个句子.
    for sentence in x_train:        # sentence: (每个)句子.
        # 3. 处理超长文本, 截断维度, 保留前cutlen个元素
        if len(sentence) > cutlen:
            # 长度超长, 就: 截断, 并添加到列表中.
            list1.append(sentence[:cutlen])
        # 4. 走到这里, 处理短序列.
        else:
            # 4.1 计算需要补齐 0 的量.
            padding_len = cutlen - len(sentence)
            # 4.2 创建补齐的列表, 并添加到列表中.
            list1.append(sentence + [0] * padding_len)

    # 5. 返回处理后的列表.
    return list1


# todo 3. 测试代码.
if __name__ == '__main__':
    # 1.定义遍历, 记录: 文本信息.
    x_train = [
        [1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
        [2, 32, 1, 23, 1]
    ]

    # 2. 调用上述的函数, 实现: 截断, 补齐.
    # result = padding(x_train)
    result = padding_custom(x_train)


    # 3. 打印处理后的内容.
    print(f'result: {result}')

    import torch.nn as nn
    nn.RNN()