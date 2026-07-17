"""
案例:
    演示传统的RNN代码实现

RNN介绍:
    概述:
        Recurrent neural network, 循环神经网络, 主要处理 序列数据 相关问题.
        序列数据: 后边的数据 对前边的数据 有依赖.
    分类:
        按照 输入 和 输出划分:
            N vs N:     输入n个, 输出n个, 适用于: 对联, 合辙诗词...
            N vs 1:     情感分析, 文本分类, 意图识别...
            1 vs N:     生成类的任务, 看图说话...
            N vs M:     Seq2Seq, 机器翻译...                # 用的最多

        按照 内部结构 划分:
            传统的RNN:
                输入层, 隐藏层(词嵌入层, 循环网络层), 输出层
            LSTM:
                遗忘门, 输入门, 细胞状态, 输出门
            Bi-LSTM:
                针对于语料, 从前往后做一次LSTM, 从后往前做一次LSTM, 然后将两个结果拼接起来.
            GRU:
                重置门, 更新门
            Bi-GRU:
                针对于语料, 从前往后做一次GRU, 从后往前做一次GRU, 然后将两个结果拼接起来.

    传统RNN的优缺点:
        优点:
            内部结构简单, 资源消耗相对较小, 适合: 处理短序列文本.
        缺点:
            处理长序列数据时, 因为 反向传播 结合 梯度连乘, 过大或者过小的w值会导致 梯度爆炸或者梯度消失.
"""

# 导包
import torch
import torch.nn as nn


# todo 1.定义函数, 演示RNN的基础版代码.
def dm_rnn_for_base():
    # 1. 创建RNN模型.
    # 参1: 词向量维度(输入维度),  参2: 隐藏层维度(输出维度),  参3: 隐藏层层数.
    rnn = nn.RNN(5, 6, 1)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length),  参2: 批次大小(batch_size),  参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(1, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers),  参2: 批次大小(batch_size),  参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行RNN模型
    # 本次的输出, 本次的隐藏状态 = rnn(本次的输入, 上一时刻的隐藏状态)
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')        # shape: (1, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')                # shape: (1, 3, 6)
    print(f'rnn模型: {rnn}')                                         # RNN(5, 6)


# todo 2.定义函数, 修改RNN模型(句子)的长度
def dm_rnn_for_sequence_len():
    # 1. 创建RNN模型.
    # 参1: 词向量维度(输入维度),  参2: 隐藏层维度(输出维度),  参3: 隐藏层层数.
    rnn = nn.RNN(5, 6, 1)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length),  参2: 批次大小(batch_size),  参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(20, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers),  参2: 批次大小(batch_size),  参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(1, 3, 6)

    # 4. 运行RNN模型
    # 本次的输出, 本次的隐藏状态 = rnn(本次的输入, 上一时刻的隐藏状态)
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')        # shape: (20, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')                # shape: (1, 3, 6)
    print(f'rnn模型: {rnn}')                                         # RNN(5, 6)


# todo 3.定义函数, 修改RNN模型 隐藏层层数
def dm_rnn_for_hidden_layers():
    # 1. 创建RNN模型.
    # 参1: 词向量维度(输入维度),  参2: 隐藏层维度(输出维度),  参3: 隐藏层层数.
    rnn = nn.RNN(5, 6, 2)

    # 2. 准备输入数据(即: 本次的输入)
    # 参1: 句子长度(sequence_length),  参2: 批次大小(batch_size),  参3: 词向量维度(即:输入维度 input_size)
    input = torch.randn(1, 3, 5)

    # 3. 初始化隐藏层(即: 上一时间步的隐藏状态)
    # 参1: 隐藏层的层数(即: 隐藏层层数 num_layers),  参2: 批次大小(batch_size),  参3: 隐藏层维度(即: 输出维度 hidden_size)
    h0 = torch.randn(2, 3, 6)

    # 4. 运行RNN模型
    # 本次的输出, 本次的隐藏状态 = rnn(本次的输入, 上一时刻的隐藏状态)
    output, hn = rnn(input, h0)

    # 5. 打印结果.
    print(f'output: {output}, output.shape: {output.shape}')        # shape: (1, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')                # shape: (2, 3, 6)
    print(f'rnn模型: {rnn}')                                         # RNN(5, 6, num_layers=2)




# todo 4.测试代码.
if __name__ == '__main__':
    # 1. 测试 RNN基础版
    # dm_rnn_for_base()

    # 2. 测试 RNN修改句子长度
    # dm_rnn_for_sequence_len()

    # 3. 测试 RNN修改隐藏层层数
    dm_rnn_for_hidden_layers()
