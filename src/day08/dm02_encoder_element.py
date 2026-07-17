"""
案例:
    演示Transformer架构的 编码器各个组件介绍.

    编码器组件的组成:
        多头注意力层 + 残差连接 + 规范化层
        前馈全连接层 + 残差连接 + 规范化层

回顾: 自注意力计算规则
    Attention(Q, K, V) = Softmax(Q * K^t / √d_k) * V

总结:
    上三角矩阵: 对角线下边是0, 上面有值.
    下三角矩阵: 对角线上面是0, 下面有值.

    Numpy#triu(m, k):  m表示1个矩阵, k: 表示对角线的起始位置(默认值是0), 返回的是: 上三角矩阵.
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from dm01_input import *


# todo 1. 测试 上三角矩阵.
def dm01_test_triu():
    # 1. 生成1个二维矩阵.
    arr = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
        [5, 5, 5, 5, 5]
    ]

    # 2. 分别测试: k=0, k=1, k=-1的情况.
    print(np.triu(arr, k=0))
    print(' -.- ' * 10)
    print(np.triu(arr, k=1))  # 对角线
    print(' -.- ' * 10)
    print(np.triu(arr, k=-1))


# todo 2. 测试 下三角矩阵.
def dm02_test_triu(size):
    # 1. 生成上三角矩阵(初始用 triu()函数来构造即可 )
    temp = np.triu(m=np.ones((1, size, size)), k=1).astype('uint8')
    print(f'temp: \n{temp}')

    # 2. 将上三角矩阵转换为下三角矩阵.
    return torch.from_numpy(1 - temp)


# todo 3. 掩码张量的可视化.
def dm03_test_mask():
    # 简单说(总结): 纵向选'当前位置', 横向看'能看哪些位置',   0(黄色): 遮挡, 1: 不遮挡.
    plt.figure(figsize=(5, 5))
    plt.imshow(dm02_test_triu(size=20)[0])
    plt.show()


# todo 4. 定义函数, 进行注意力的计算.
def attention(query, key, value, mask=None, dropout=None):
    """
    函数功能: 自定义代码, 模拟: 注意力计算.
    :param query: 查询张量, 形状通常是: [batch_size, seq_len, d_model]
    :param key: 键张量, 形状通常是: [batch_size, seq_len, d_model]
    :param value: 值张量, 形状通常是: [batch_size, seq_len, d_model]
    :param mask: 掩码张量, 形状一般和 scores匹配.
    :param dropout: 随机失活, 防止过拟合.
    :return:
    """
    # 1. 定义查询张量Q的特征维度.
    # 例如: query的形状是[2, 4, 512] 每批2句话, 每句话有4个单词, 每个单词的维度是512.  d_k = 512
    d_k = query.size(-1)

    # 2. 计算原始注意力分数, 即: Q * K^t / √d_k
    # key.transpose(-2, -1): 从原来的维度[batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 3. 掩码处理(可选)
    if mask is not None:
        # 将scores中, 遮挡的部分, 设置为负无穷(极小值), 这样softmax()后这些位置的权重会接近0(对应: Decoder中'不能看未来位置'的需求)
        scores = scores.masked_fill(mask == 0, -1e9)

    # 4. 计算注意力权重.
    # dim=-1, 在最后1个维度上进行归一化, 让每个query位置的权重总和为 1
    p_attn = F.softmax(scores, dim=-1)

    # 5. 随机失活(可选)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 6. 计算最终的注意力输出: 权重加权求和.
    # 返回两个结果: 注意力输出(融合信息) 和 注意力权重(用于可视化,调试)
    return torch.matmul(p_attn, value), p_attn


# todo 5. 测试注意力机制.
def use_attention():
    # 1. 获取位置编码处理后的结果(词嵌入层结果 + 位置编码结果)
    position_x = use_position()

    # 2. 因为是自注意力机制, Q,K,V都用同一个张量.
    query = key = value = position_x

    # 3. 没有掩码, 调用 attention()
    result1, p_attn = attention(query, key, value)
    print(f'result1: {result1.shape}')      # 注意力输出张量的形状: [batch_size, seq_len, d_model] -> [2, 4, 512]
    print(f'p_attn: {p_attn.shape}')        # 注意力权重张量的形状: [batch_size, seq_len, seq_len] -> [2, 4, 4], 表示每个词对其他词的权重.
    print(' -.- ' * 10 )

    # 4. 构造1个全0的掩码张量(简单玩儿玩儿, 看看效果即可)
    mask = torch.zeros(2, 4, 4)
    # 带掩码, 调用 attention()
    result2, p_attn = attention(query, key, value, mask)
    print(f'result2: {result2.shape}')
    print(f'p_attn: {p_attn.shape}')


# todo 6. 定义1个克隆函数 -> 实现克隆指定的模块N次, 返回N个相同模块的 列表.
def clones(module, N):
    """
    函数功能: 创建N个相同的模块, 返回一个列表. 深拷贝, 每个模块拥有独立的参数.
    :param module: 要被克隆的模块.
    :param N: 克隆的次数.
    :return: 具有N个相同模块的列表
    """
    # return nn.ModuleList([具体的深拷贝动作 for _ in range(N)])
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# todo 7. 定义函数, 实现多头注意力机制 -> 核心: 把词向量维度分成N个头, 并行计算'小维度'的注意力, 增强模型表达能力.
class MultiHeadAttention(nn.Module):
    # 1. 参数初始化.
    # 参1: 词向量维度(例如: 512), 参2: 多头数量(例如: 8), 参3: 随机失活概率(例如: 0.1)
    def __init__(self, embed_dim, head, dropout_p=0.1):
        # 1. 初始化父类成员.
        super().__init__()

        # 2. 分头动作, 确保能整除.
        assert embed_dim % head == 0        # 只有 embed_dim % head == 0 条件成立, 代码才会继续向下执行.

        # 3. 计算每个头的词嵌入维度.
        self.d_k = embed_dim // head        # 512 // 8 = 64
        self.head = head                    # 8

        # 4. 定义4个线性层, 前3个负责: Q,K,V的投影(映射), 最后1个负责: 用于输出(多头注意力机制)的结果.
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)

        # 5. 定义随机失活层.
        self.dropout = nn.Dropout(p=dropout_p)
        # 6. 保存注意力权重, 用于可视化 或者 分析.
        self.attn = None


    # 2. 前向传播 -> 实现多头注意力计算流程.
    def forward(self, query, key, value, mask=None):
        # 1. 判断是否要掩码.
        # Q, K, V的形状: [batch_size, seq_len, d_model], 例如: [2, 4, 512]
        # mask的形状: [batch_size, seq_len, seq_len], 例如: [1, 4, 4]
        if mask is not None:
            mask = mask.unsqueeze(0)

        # 2. 获取batch_size的批量大小,
        self.batch = query.size(0)

        # 3. 线性变换: Q, K, V -> 多头注意力计算.
        # mode(x):          通过线性层 将输入映射到 embed_dim的维度, 即: 512
        # view(...):        把输入张量转成 [batch_size, seq_len, head, d_k] -> [2, 4, 8, 64]   ==> [batch_size, head, seq_len, d_k] -> [2, 8, 4, 64]
        # transpose(1, 2):   把 [batch_size, seq_len, head, d_k] -> [batch_size, head, seq_len, d_k]
        query, key, value = [
            model(x).view(self.batch, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]

        # 4. 多头注意力计算
        # x的形状(注意力输出):      [batch_size, head, seq_len, d_k] -> [2, 8, 4, 64]
        # self.att(注意力权重):    [batch_size, head, seq_len, seq_len] -> [2, 8, 4, 4]
        x, self.attn = attention(query, key, value, mask, self.dropout)

        # 5. 将多头注意力结果 -> 合并.
        # 例如: [2, 8, 4, 64] -> [2, 4, 8, 64] -> [2, 4, 512]
        attn_x= x.transpose(1, 2).contiguous().view(self.batch, -1, self.head * self.d_k)

        # 6. 通过最后1个线性层处理, 返回结果.
        return self.linears[-1](attn_x)


# todo 8. 测试多头注意力机制.
def use_multihead():
    # 1. 创建多头注意力层 对象.
    my_attention = MultiHeadAttention(512, 8)
    # 2. 获取位置编码处理后的结果(词嵌入层结果 + 位置编码结果)
    position_x = use_position()
    # 3. 定义遍历, 表示Q, K, V, 因为是自注意力机制, Q,K,V都用同一个张量.
    query = key = value = position_x
    # 4. 创建掩码张量, 形状是: [batch_size, seq_len, seq_len]
    mask = torch.zeros(8, 4, 4)
    # 5. 将数据送给模型 -> 获取多头注意力结果.
    result = my_attention(query, key, value, mask)
    # 6. 打印输出形状, 要与输入的形状保持一致, 即: [2, 4, 512]
    print(f'(多头注意力层)result: {result.shape}')

    # 7. 返回多头注意力结果, 后续可以直接用.
    return result


# todo 9. 初始化前馈全连接层(它包含2个全连接层) -> 对注意力层的结果进一步加工(强化特征)
class FeedForward(nn.Module):
    # 1. 初始化函数
    # 参1: 输入数据的词向量维度(例如: 512), 参2: 前馈全连接层维度(例如: 2048), 参3: 随机失活概率(例如: 0.1)
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 定义第1个全连接层, 输入维度: d_model, 输出维度: d_ff
        self.linear1 = nn.Linear(d_model, d_ff)     # 512维 -> 2048维
        # 3. 定义第2个全连接层, 输入维度: d_ff, 输出维度: d_model
        self.linear2 = nn.Linear(d_ff, d_model)     # 2048维 -> 512维
        # 4. 创建随机失活层.
        self.dropout = nn.Dropout(p=dropout_p)

    # 2. 前向传播 -> 实现前馈全连接层的计算流程.
    def forward(self, x):
        # 1. 通过第1个全连接层, 映射到 d_ff的维度.
        x = self.linear1(x)
        # 2. 通过激活函数处理(Relu会把负值变为0, 保留正数, 增加了非线性功能) -> 通过随机失活处理.
        x = self.dropout(F.relu(x))
        # 3. 通过第2个全连接层, 映射到 d_model的维度.
        x = self.linear2(x)
        # 4. 返回结果.
        return x


# todo 10. 测试前馈全连接层 -> 先拿到多头注意力层结果, 把结果送给前馈全连接层处理.
def use_ff():
    # 1. 获取多头注意力层结果.
    attn_x = use_multihead()        # 形状: [2, 4, 512]
    # 2. 创建前馈全连接层对象.
    my_ff = FeedForward(512, 2048)
    # 3. 将多头注意力层结果, 送给前馈全连接层处理.
    result = my_ff(attn_x)
    # 4. 输出结果.
    print(f'(前馈全连接层)result: {result.shape}')        # torch.Size([2, 4, 512])
    # 5. 返回结果.
    return result



# todo n.测试代码
if __name__ == '__main__':
    # 1. 测试 上三角矩阵.
    # dm01_test_triu()

    # 2. 测试 下三角矩阵.
    # print(dm02_test_triu(size=5))
    # print(dm02_test_triu(size=5)[0])

    # 3. 测试: 掩码张量的可视化.
    # dm03_test_mask()

    # 4. 测试: 注意力机制.
    # use_attention()

    # 5. 测试: 多头注意力机制.
    # use_multihead()

    # 6. 测试: 前馈全连接层.
    use_ff()