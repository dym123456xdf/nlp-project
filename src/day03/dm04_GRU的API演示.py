"""
案例:
    演示GRU代码

GRU简介:
    概述:
        Gated Recurrent Unit, 门控循环单元结构, 可以理解为是一个 简化版的智能管家,
        即能像LSTM那样处理长序列信息, 计算上也更简单, 更节省资源.
    核心设计思路:
        重置门, 更新门.       就是在LSTM的基础上, 对部分门做合并.
    优点:
        和LSTM一样, 既可以处理长序列语义关联, 能够有效缓解梯度消失和爆炸问题.
        关于长序列的处理效果: 优于传统RNN
        关于计算复杂度: 比LSTM要小.
    缺点:
        不能完全解决梯度消失, 爆炸问题, 不能并行计算.
        在数据量和模型体量逐步增大的未来, 是RNN发展的关键瓶颈.
"""

# 导包
import torch
import torch.nn as nn


# todo 1.定义函数, 演示: GRU代码实现.
def gru_demo():
    # 1. 创建GRU模型对象.
    # 参1: 输入特征维度(词向量维度), 参2: 隐藏层维度(输出维度), 参3: 层数
    gru = nn.GRU(input_size=5, hidden_size=6, num_layers=1)

    # 2. 创建输入数据.
    # 参1: 句子长度(seq_len), 参2: 批次大小(batch_size), 参3: 输入特征维度(input_size)
    input = torch.randn(2, 3, 5)

    # 3. 创建初始隐藏状态.
    # 参1: 层数(num_layers), 参2: 批次大小(batch_size), 参3: 隐藏层维度(hidden_size)
    h0 = torch.zeros(1, 3, 6)

    # 4. 运行GRU模型.
    output, hn = gru(input, h0)

    # 5. 输出结果.
    print(f'output: {output}, output.shape: {output.shape}')  # shape: (2, 3, 6)
    print(f'hidden: {hn}, hidden.shape: {hn.shape}')          # shape: (1, 3, 6)
    print(f'gru模型: {gru}')                                   # GRU(5, 6)


# todo 2. 测试代码
if __name__ == '__main__':
    gru_demo()