"""
案例:
    演示解码器层的代码实现.

解码器层由3个子层组成:
    子层1: Masked Multi-Head Attention, 掩码多头注意力机制层 + 残差连接 + 规范化层
    子层2: Multi-Head Attention, 多头注意力机制层 + 残差连接 + 规范化层
    子层3: Feed Forward, 前向传播层(前馈全连接层) + 残差连接 + 规范化层
"""

# 导包
from dm05_encoder import *
import copy
import math


# todo 1. 定义解码器层 -> 由3个子层组成.
class DecoderLayer(nn.Module):
    # 1. 初始化函数.
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout=0.1):
        """
        初始化函数, 给变量赋值的.
        :param d_model: 词向量维度.
        :param self_attn: 注意力机制, 处理(解码器输入序列内部关系), 即: 解码器的输入.
        :param src_attn: 源序列(编码器-解码器)注意力机制. )
        :param feed_forward: 前向传播层, 强化特征的.
        :param dropout: 随机失活概率.
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 定义属性.
        self.d_model = d_model
        self.self_attn = self_attn      # 解码器层的 自注意力机制层(掩码多头注意力机制层)
        self.src_attn = src_attn        # 解码器层的 源序列(编码器-解码器) 注意力机制层
        self.feed_forward = feed_forward
        # 3. 定义3个子层连接结构.
        self.layers = clones(SublayerConnection(d_model, dropout), 3)   # [第1个子层对象, 第2个子层对象, 第3个子层对象]


    # 2. 解码器层的 前向传播.
    def forward(self, x, encoder_output, source_mask, target_mask):
        """
        解码器层的 前向传播动作.
        :param x: 解码器的输入序列(即: 词嵌入 + 位置编码)
        :param encoder_output: 编码器的输出序列(词嵌入 + 源序列位置编码)
        :param source_mask: 源序列的填充掩码, 用于 编码器-解码器 注意力.
        :param target_mask: 目标序列的填充掩码, 用于: 自注意力.
        :return:
        """
        # 1. 经过第1个子层 -> 多头自注意力机制层(掩码多头注意力机制)
        x = self.layers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 2. 经过第2个子层 -> 多头注意力机制层(编码器-解码器)
        x = self.layers[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, source_mask))
        # 3. 经过第3个子层 -> 前向传播层(前馈全连接层)
        x = self.layers[2](x, self.feed_forward)
        # 4. 返回结果
        return x


# todo 2. 测试解码器层.
def use_decoder_layer():
    # 1. 定义变量, 记录: 解码器的输入 -> Q
    y = torch.LongTensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])  # 2个句子, 每个句子4个词.

    # 2. 将值传入到Embedding层(词嵌入层)
    # 2.1 创建词嵌入层对象.
    my_embed = Embedding(1000, 512)     # 词表大小: 1000, 词嵌入维度: 512
    # 2.2 把上述的y(输入) -> 词向量形式.
    embed_y = my_embed(y)
    print(f'词嵌入层: {embed_y.shape}')     # [2, 4, 512]

    # 3. 将 embed_y(词嵌入处理结果) -> 位置编码.
    my_position = PositionalEncoding(512, 0.1)
    position_y = my_position(embed_y)
    print(f'位置编码层: {position_y}')       # [2, 4, 512]

    # 4.创建多头注意力层
    multi_attn = MultiHeadAttention(512, 8)
    self_attn = copy.deepcopy(multi_attn)       # 深拷贝 -> 确保参数不会共享.
    src_attn = copy.deepcopy(multi_attn)

    # 5. 创建前向传播层(前馈全连接层)
    ff = FeedForward(512, 2048)
    # 6. 获取编码器的输出结果.
    encoder_output = use_encoder()

    # 7. 定义mask(掩码矩阵): source_mask 和 target_mask真实作用不同(要根据你真正的业务来判断), 这里我就随便举例了.
    source_mask = torch.zeros(8, 4, 4)
    target_mask = torch.zeros(8, 4, 4)

    # 8. 实例化 解码器层对象
    my_decoder_layer = DecoderLayer(512, self_attn, src_attn, ff)
    # 9. 将数据传给解码器层, 获取其输出结果.
    result = my_decoder_layer(position_y, encoder_output, source_mask, target_mask)

    # 10. 打印结果.
    print(f'解码器层的形状: {result.shape}')      # [2, 4, 512]
    print(f'解码器层对象: \n{my_decoder_layer}')



# todo 3. 测试代码.
if __name__ == '__main__':
    use_decoder_layer()
