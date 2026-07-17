"""
案例:
    演示文件张量实现方式之 word Embedding.

大白话解释下: word2vec 和 word Embedding的区别
    word2vec:
        先办证, 后干活. 会预先训练处 词向量(模型), 后续再带入模型做其它操作.
    word Embedding:
        边办证, 边干活. 训练过程中, 词向量会自动生成. -> 就是以前我们学RNN的 词嵌入层.

细节: 如果你要看TensorBoard的可视化界面(本质就是一个 PCA主成分分析图), 代码写完后, 按照如下操作即可.
    step1: 切换到 nlpbase 沙箱
    step2: 切换到当前项目(这里是day02)的路径下, 即: 有 runs文件夹的那个目录下.
    step3: 运行如下的命令即可
        tensorboard --logdir=runs --host 0.0.0.0
    step4: 通过 localhost:6006 访问即可.
        注意: 第一次加载会慢一点, 你稍微等会儿.
"""

# 禁用oneDNN优化包...
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 导包
import torch                                                    # 深度学习框架, 封装了和张量相关的各种操作
from tensorflow.keras.preprocessing.text import Tokenizer       # 词汇映射器.
from torch.utils.tensorboard import SummaryWriter               # 可视化 词向量
import jieba                                                    # 分词器.
import torch.nn as nn                                           # 神经网络模块


# todo 1. 定义函数, 演示: word Embedding将文本转成词向量, 并可视化.
def dm01_embedding_show():
    # 1. 定义遍历, 记录: 待处理的文本.
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"

    # 2. 把上述的两句话, 封装成: 列表.
    sentences = [sentence1, sentence2]

    # 3. 使用jieba进行分词处理.
    # 3.1 定义遍历, 记录: 分词后的数据 -> 即: 词语列表
    word_list = []
    # 3.2 遍历上述的句子列表, 获取到每个句子.
    for sentence in sentences:
        # 3.3 使用jieba进行分词处理, 并将分词结果添加到 word_list 中.
        word_list.append(jieba.lcut(sentence))
    # 3.4 打印分词结果.
    """
        [
            ['传智', '教育', '是', '一家', '上市公司', '，', '旗下', '有', '黑马', '程序员', '品牌', '。', '我', '是', '在', '黑马', '这里', '学习', '人工智能'], 
            ['我', '爱', '自然语言', '处理']
        ]
    """
    print(f'分词结果: {word_list}')


    # 4.构建词汇表, 进行 文本数值化(词向量)
    # 4.1 初始化词汇映射器.
    my_tokenizer = Tokenizer()
    # 4.2 拟合训练数据, 统计词频, 并构建: 词汇表
    my_tokenizer.fit_on_texts(word_list)
    # 4.3 查看 词 和 索引的映射关系.
    # {'是': 1, '黑马': 2, '我': 3, '传智': 4, '教育': 5, '一家': 6, '上市公司': 7, '，': 8, '旗下': 9, '有': 10, '程序员': 11, '品牌': 12, '。': 13, '在': 14, '这里': 15, '学习': 16, '人工智能': 17, '爱': 18, '自然语言': 19, '处理': 20}
    print(f'词和索引的映射关系: {my_tokenizer.word_index}')

    # 4.4 获取去重后的 所有词汇列表.
    my_token_list = my_tokenizer.word_index.values()
    print(my_token_list)             # dict_values([1, 2, 3,..., 20])

    # 4.5 将分词后的文本 -> 转成 数字序列.
    seq2id = my_tokenizer.texts_to_sequences(word_list)
    print(f'文本转成数字序列: {seq2id}')        # [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [3, 18, 19, 20]]


    # 5. 创建 词嵌入层, 把文本(即: 词对应的编号) 转成 词向量.
    # 5.1 创建 词嵌入层对象.
    # 参1: 词汇表大小, 即: 唯一的词的个数.        参2: 词向量的维度
    embed = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    # 5.2 查看 词嵌入层的 权重参数(即: 词向量)
    print(f'embed: {embed.weight.data}')
    print(f'embed.shape: {embed.weight.data.shape}')        # torch.Size([20, 8])
    print(' =.= ' * 10)

    # 6. 词向量可视化
    # 6.1 创建TensorBoard写入器, 将数据写入到 runs 目录
    my_summary = SummaryWriter(log_dir='./runs')
    # 6.2 将词向量 和 对应的词语 添加到 TensorBoard中
    # 参1: 词向量矩阵, 形状是: (20, 8), 20个词, 每个词用8个数字表示(8维)
    # 参2: 对应的词语列表, 用来标注每个点.
    my_summary.add_embedding(embed.weight.data, my_token_list)
    # 6.3 关闭写入器
    my_summary.close()

    # 7. 查看每个单词对应的词向量.
    for idx in range(len(my_tokenizer.word_index)): # idx的范围: [0, 20)
        # 7.1 获取当前单词对应的词向量.
        temp_vector = embed(torch.tensor(idx))
        # print(f'词向量: {temp_vector}')       # tensor([ 0.1227,  0.6931,  0.0187,  1.3011,  0.1021,  2.4951,  0.8910, -0.1551], grad_fn=<EmbeddingBackward0>)

        # 7.2 获取当前索引对应的单词, my_tokenizer.index_word的索引是从1开始的.
        word = my_tokenizer.index_word[idx + 1]
        print(f'单词: {word}, 词向量: {temp_vector.detach().numpy()}')


# todo 2. 测试代码
if __name__ == '__main__':
    dm01_embedding_show()