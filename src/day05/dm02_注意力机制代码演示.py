"""
案例:
    演示注意力机制的计算流程.

任务描述:
    已知QKV：v是内容比如32个单词，每个单词64个特征，k是32个单词的索引，q是查询张量
我们的任务：输入查询张量q，通过注意力机制来计算如下信息：
    1、查询张量q的注意力权重分布：查询张量q和32个单词相关性（相识度）
    2、查询张量q的结果表示：由一个普通的q升级成一个更强大q
    3、注意：查询张量q查询的目标是谁，就是谁的查询张量。
       例如: 查询张量q是来查询单词"我"，则q就是"我"的查询张量


MyAttention类实现思路分析:
    1. 初始化参数 __init__()魔法方法
        query_size, key_size, value_size1, value_size2, output_size
        准备2个线性层 注意力权重分布self.attn  注意力结果表示: 按照指定维度进行输出 self.attn_combine

    2. forward(self, Q, K, V)
        求查询张量Q的注意力权重分布, attn_weights[1, 32]
        求查询张量q的注意结果表示 bmm运算, attn_applied[1, 1, 64]
        q 和 attn_applied融合, 再按照指定维度输出 output[1, 1, 32]
        返回注意力结果表示 output: [1, 1, 32], 注意力权重分布 attn_weights: [1, 32]
"""

# 导包
import torch
import torch.nn as nn
import torch.nn.functional as F


# todo 1.自定义注意力机制模块, 实现: Q(查询张量) 和 键值对(K, V)的注意力计算.
class MyAttn(nn.Module):            # Attention: 注意力.
    # todo 1.1 初始化函数.
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        """
        初始化函数, 用于初始化 注意力机制的核心参数.
        :param query_size: 查询张量的维度
        :param key_size:   键张量的维度
        :param value_size1: 值张量的序列长度(即: 有多少个单词)
        :param value_size2: 值张量的维度(词向量), 例如: 64
        :param output_size: 输出张量的维度
        """
        # 1. 初始化父类的内容.
        super().__init__()

        # 2. 定义参数.
        self.query_size = query_size        # 查询张量Q的维度, 例如: 32
        self.key_size = key_size            # 键张量K的维度, 例如: 32
        self.value_size1 = value_size1      # 值张量V的序列长度, 例如: 32
        self.value_size2 = value_size2      # 值张量V的维度, 例如: 64
        self.output_size = output_size      # 输出张量的维度, 例如: 32

        # 3. 注意力权重计算层: 将Q和K拼接后, 映射到: 值序列长度维度.
        # 例如: 输入维度Q(32) + K(32) = 64,  输出维度: 32
        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)

        # 4. 注意力融合层: 将原始Q 和 注意力加权后的V拼接(融合)后, 映射到: 输出维度.
        # 例如: 输入维度Q(32) + V(64) = 96,  输出维度: 32
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)



    # todo 1.2 前向传播, 计算: 注意力权重 和 输出.
    def forward(self, Q, K, V):
        """
        前向传播, 计算: 注意力权重 和 (最终)输出.
        :param Q: 查询张量, 形状: [1, 1, query_size] -> [1, 1, 32]
        :param K: 键张量, 形状: [1, 1, key_size] -> [1, 1, 32]
        :param V: 值张量, 形状: [1, value_size1, value_size2] -> [1, 32, 64]
        :return:
        """
        # 1. 计算注意力权重
        # 1.1 拼接Q和K, 维度变化: [1, 1, 32] + 1, 1, 32] = [1, 1, 64]
        qk_cat = torch.cat((Q[0], K[0]), dim=-1)
        print(f'qk_cat的形状: {qk_cat.shape}')

        # 1.2 通过线性层计算注意力得分, 维度变化: [1, 64] -> [1,  32]
        attn_scores = self.attn(qk_cat)
        print(f'attn_scores的形状: {attn_scores.shape}')

        # 1.3 用softmax()将得分 -> 概率分布:[1, 32] ->  [1, 32]
        attn_weights = F.softmax(attn_scores, dim=-1)
        print(f'attn_weights的形状: {attn_weights.shape}')

        # 2. 应用注意力权重(注意力分配系数) 到 值张量V
        # 2.1 扩展注意力权重维度, 以便匹配v的批次维度, 即: [1, 32] -> [1, 1, 32]
        attn_weights_expanded = attn_weights.unsqueeze(0)

        # 2.2 使用bmm()函数执行矩阵乘法, 维度变化: [1, 1, 32] * [1, 32, 64] = [1, 1, 64]
        attn_applied = torch.bmm(attn_weights_expanded, V)
        print(f'attn_applied的形状: {attn_applied.shape}')


        # 3. 融合原始查询Q 和 注意力加权后的V
        # 3.1 拼接Q和V, 维度变化: [1, 1, 32] + [1, 1, 64] = [1, 1, 96]
        output_cat = torch.cat((Q, attn_applied), dim=-1)
        print(f'output_cat的形状: {output_cat.shape}')

        # 3.2 通过线性层降维到输出维度.
        # 维度变化: [1, 96] -> [1, 32] -> [1, 1, 32]
        output = self.attn_combine(output_cat)
        print(f'output的形状: {output.shape}')

        # 4. 返回结果
        return output, attn_weights         # output: [1, 1, 32], attn_weights: [1, 32]


# todo 2. 测试代码
if __name__ == '__main__':
    # 1. 实例化参数设置.
    query_size, key_size, value_size1, value_size2, output_size = 32, 32, 32, 64, 32

    # 2. 创建随机输入张量.
    Q = torch.randn(1, 1, query_size)               # 查询张量: [批次, 序列, 特征] -> [1, 1, 32]torch.Size([1, 32])
    K = torch.randn(1, 1, key_size)                 # 键张量: [批次, 序列, 特征] -> [1, 1, 32]
    V = torch.randn(1, value_size1, value_size2)    # 值张量: [批次, 单词数, 词向量维度] -> [1, 32, 64]

    # 3. 实例化自定义的注意力机制模块, 并测试.
    my_attn = MyAttn(query_size, key_size, value_size1, value_size2, output_size)
    output, attn_weights = my_attn(Q, K, V)

    # 4. 输出结果.
    print(' =.= ' * 40)
    print(f'查询张量Q的注意力结果表示: {output.shape}, {output}')               # torch.Size([1, 1, 32])
    print(f'查询张量Q的注意力权重分布: {attn_weights.shape}, {attn_weights}')   # torch.Size([1, 32])