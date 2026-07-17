"""
案例:
    演示Transformer架构中的输入部分 -> 词嵌入层 和 位置编码.

总结(回顾):
    Transformer的输入部分由2部分组成, 分别是:
        词嵌入层(Word Embedding)
        位置编码(Positional Encoding)
"""

# 导包
import torch
import torch.nn as nn   # neural network: 神经网络
import numpy as np
import math
import matplotlib.pyplot as plt


# todo 1. 定义类(模拟词嵌入层), 实现输入部分 -> 词嵌入层 的功能.
class Embedding(nn.Module):
    # 1. 初始化函数.
    def __init__(self, vocab_size, d_model):
        """
        该函数目的: 初始化参数用的
        :param vocab_size: 词汇表大小(去重后的单词个数)
        :param d_model: 词嵌入的维度.
        """
        # 1. 初始化父类的信息.
        super().__init__()
        # 2. 定义遍历, 接收: 词汇表大小(去重后的单词个数), 词嵌入的维度.
        self.vocab_size = vocab_size
        self.d_model = d_model

        # 3. 定义词嵌入层, 将单词索引映射为对应的词嵌入向量.
        # '欢迎来武汉' -> {0: '欢迎', 1: '来', 2: '武汉'} -> 把索引(0, 1, 2)转成词向量形式.
        self.embed = nn.Embedding(vocab_size, d_model)


    # 2. 前向传播函数.
    def forward(self, x):
        # 将输入的单词索引映射为词向量, 并乘以 根号d_model 进行缩放.
        # 缩放的目的: 为了平衡梯度, 避免梯度消失 或者 梯度爆炸.
        return self.embed(x) * math.sqrt(self.d_model)


# todo 2. 测试Embedding(词嵌入层)
def dm01_embedding():
    # 1. 定义遍历, 记录: 词表大小, 词嵌入维度.
    vocab_size, d_model = 1000, 512     # 1000个单词, 每个单词的维度为512.

    # 2. 创建(自定义的)词嵌入层对象.
    my_embed = Embedding(vocab_size, d_model)

    # 3. 创建张量, 包含2个句子, 每个句子4个单词.
    x = torch.tensor([
        # (例如)单词
        # ['我', '爱', '吃', '汤圆'],
        # ['我', '爱', '吃', '蛋酒'],

        # 单词索引
        [100, 2, 421, 300],
        [500, 888, 306, 509]
    ])

    # 4. 前向传播, 获取词嵌入向量.
    result = my_embed(x)

    # 5. 输出结果.
    print(f'result: {result}, result.shape: {result.shape}')        # shape: torch.Size([2, 4, 512])


# todo 3.定义类(模拟位置编码层), 实现输入部分 -> 位置编码 的功能.
class PositionalEncoding(nn.Module):
    # 1. 初始化函数.
    def __init__(self, d_model, dropout, max_len=60):
        """
        该函数目的: 初始化参数用的
        :param d_model: 词向量维度, 例如; 512
        :param dropout: 随机失活概率
        :param max_len: 最大句子长度
        """
        # 1. 初始化父类信息.
        super().__init__()
        # 2. 定义dropout层, 防止: 过拟合.
        self.dropout = nn.Dropout(p=dropout)
        # 3. 定义pe(Positional Encoding), 用于保存位置编码结果.
        pe = torch.zeros(max_len, d_model)  # shape: [60个词, 512维]
        # 4. 定义1个位置列向量, 范围: 0 ~ max_len - 1
        position = torch.arange(0, max_len).unsqueeze(1)    # shape: [60个词, 1维]
        # print(f'position:{ position}, position.shape: {position.shape}')

        # 5. 定义1个变化矩阵, 本质是: 公式里的 1 / 10000^(2i / d_model)
        # 10000 ^ (2i / d_model) = e ^ ((2i / d_model) * ln(10000))
        # 1 / 上述的内容, 所以求倒数: = e ^ ((2i / d_model) * -ln(10000)) -> e ^ (2i * -ln(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 形状: [1, 256]
        # print(f'div_term: {div_term.shape}')     # [256]

        # 6. 计算三角函数里边的值.
        # position形状: [max_len, 1],   div_term形状: [1, 256],   position * div_term形状: [max_len, 256]
        position_value = position * div_term
        # print(f'position_value: {position_value.shape}')    # torch.Size([60, 256])


        # 7. 进行pe的赋值, 偶数位置使用 正弦函数(sin)
        pe[:, 0::2] = torch.sin(position_value)
        # 8. 进行pe的赋值, 奇数位置使用 余弦函数(cos)
        pe[:, 1::2] = torch.cos(position_value)

        # 9. 将pe进行升维, 形状: [1, 60, 512]
        pe = pe.unsqueeze(0)        # 位置编码.
        # print(f'pe: {pe.shape}')

        # 10. 把pe注册到模型的缓冲区, 利用它, 不断的更新参数.
        self.register_buffer('pe', pe)


    # 2. 前向传播
    def forward(self, x):
        # x: 词向量, 形状: [batch_size, seq_len, d_model] -> [2, 4, 512]
        # self.pe: 位置编码信息, 形状: torch.Size([1, 60, 512]
        # 这个代码的核心是: 把 '词向量' 和 '位置编码' 进行相加(融合)
        x = x + self.pe[:, :x.size(1)]
        # print(f'x.shape: {x.shape}')                                        # [2, 4, 512], 词向量
        # print(f'self.pe[:, :x.size(1)]: {self.pe[:, :x.size(1)].shape}')    # [1, 4, 512], 位置编码信息.

        # 随机失活, 不改变形状.
        return self.dropout(x)


# todo 4. 测试PositionalEncoding(位置编码层)
def use_position():
    # 1. 定义词汇表大小 和 词嵌入维度.
    vocab_size, d_model = 1000, 512

    # 2. 实例化Embedding层.
    my_embed = Embedding(vocab_size, d_model)       # [1000, 512]

    # 3. 创建输入张量, 形状是: [batch_size, seq_len], 例如: [2, 4], 2个句子, 每个句子4个单词.
    x = torch.tensor([
        # 单词索引
        [100, 2, 421, 300],
        [500, 888, 306, 509]
    ])

    # 4. 计算词向量结果.
    embed_x = my_embed(x)
    # print(f'embed_x: {embed_x}, embed_x.shape: {embed_x.shape}')    # 词向量结果: torch.Size([2, 4, 512])

    # 5. 创建位置编码层对象.
    my_position = PositionalEncoding(d_model=d_model, dropout=0.1)

    # 6. 计算位置编码结果.
    position_x = my_position(embed_x)       # [2, 4, 512]

    # 7. 返回结果.
    return position_x


# todo n. 测试代码
if __name__ == '__main__':
    # 1. 测试词嵌入
    # dm01_embedding()

    # 2. 测试位置编码
    result = use_position()
    print(f'result: {result}, result.shape: {result.shape}')    # torch.Size([2, 4, 512])  结合了词向量和位置编码信息的最终结果.