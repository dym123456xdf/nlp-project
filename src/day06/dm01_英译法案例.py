"""
案例:
    演示 seq2seq框架(句子到句子模型), 即: RNN的 N vs M框架, 可实现 机器翻译(英译法)

基于GRU的seq2seq模型架构实现翻译的过程:
    step1: 导入工具包 和 工具函数
    step2: 对持久化文件中数据进行预处理, 以满足模型训练要求
    step3: 构建基于GRU的编码器和解码器
    step4: 构建模型训练函数, 并进行训练
    step5: 构建模型评估函数, 并进行测试以及Attention效果分析.

扩展: 如何设置PyTorch程序在GPU上运行:
    1. windows, Linux, Mac(M芯片)均可,   windows(无独显), Mac(Intel芯片)不行.
    2. 去dos窗口查看你本机的 CUDA版本.
        输入 nvidia-smi 即可
    3. 去PyTorch官网找 对应你cuda版本的安装命令
        PyTorch官网: https://pytorch.org/
        例如: 安装命令为: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    4. 推荐新建1个新的沙箱, 去里边玩儿.
        conda create -n nlpbase2
        conda activate nlpbase2
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

        conda remove -n 沙箱名 --all       这个是删除沙箱, 如果操作失败, 就删除重来.
    5. 在PyCharm中关联新的沙箱即可, 运行下边的测试代码.
    6. 如果不OK, 大概率 显卡驱动(CUDA版本) 和 你的PyTorch不匹配, 建议: 直接更新驱动.
"""

# 导包
import re               # 正则表达式相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm   # 进度条


# 设备选择, 我们可以选择在cuda 或者 cpu上运行你的代码.
# windows写法.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# mac写法.
# device = 'mps'

print(f'当前设备: {device}')


# todo 1.指定特殊的token
# 起始标记
SOS_token = 0       # start of  sentence
# 结束标记
EOS_token = 1       # end of  sentence
# 最大句子长度不能超过10(包括标点)
MAX_LENGTH = 10
# 数据文件的路径
data_path = './data/eng-fra-v2.txt'


# todo 2.定义数据清洗函数 -> 即: 字符串规范化处理函数, 用于: 文本预处理.
def normalizeString(s):
    """
    字符串规范化函数
    :param s: 需要处理的字符串
    :return: 处理后的字符串
    """
    # 1. 将字符串转成小写形式, 并去除收尾空白字符.
    s = s.lower().strip()

    # 2. 在 .!? 前加1个空格, 使用正则表达式的捕获组替换.
    # 参1: 正则表达式(即: 要被替换的内容), 参2: (用来)替换的内容, 参3: 要操作的字符串
    s = re.sub('([.!?])', r' \1', s)
    # print(f's: {s}')

    # 3. 过滤非标准字符: 保留大小写字母 和 基本的标点符号, 其它字符替换为: 空格.
    # [^a-zA-Z.!?]解释:  除了大小写字母, .!? 符号之外, 任意的1个字符
    # +解释: 数量词, 代表前边的内容至少出现1次, 至多出现无数次.
    s = re.sub('[^a-zA-Z.!?]+', ' ', s)
    # print(f's: {s}')

    # 4. 返回处理后的字符串.
    return s


# todo 3.数据预处理 -> 清洗文本 和 构建文本字典.
def my_getdata():
    # 1. 读取原始文件数据.
    # 1.1 打开文件, 使用 with open语法读取.
    with open(data_path, 'r', encoding='utf-8') as src_f:
        # 1.2 一次性读取所有行.
        lines = src_f.readlines()       # 格式为: ['第1行\n', '第2行\n', ...]
        # print(lines[:5])

        # 2. 清洗文本并构建双语 句子树(句子对).
        # for line in lines:         获取到每行数据,     例如: 'i m tired .	je suis fatigue !'
        # for s in line.split('\t'): 每行数据按照\t切割, 例如: ['i m tired .', 'je suis fatigue !']
        my_pairs = [ [normalizeString(s) for s in line.split('\t')] for line in lines]
        # print(len(my_pairs))        # 句子总对数: 63594

        # 3. 数据探查: 查看部分样本数据.
        # # 3.1 打印前5条双语句子对, 直观查看数据格式.
        # print(f'my_pairs: {my_pairs[:5]}')
        #
        # # 3.2 打印第7000条双语句子对, 查看特定位置的数据.
        # print(f'my_pairs[7000]: {my_pairs[7000]}')
        # print(f'my_pairs[7000](英文): {my_pairs[7000][0]}')
        # print(f'my_pairs[7000](法文): {my_pairs[7000][1]}')

        # 4. 初始化英语词汇表
        # 4.1 创建单词到索引的字典映射, 预定义特殊字符SOS(句子开始), EOS(句子结束)的索引为: 0, 1
        english_word2index = {'SOS': 0, 'EOS': 1}
        # english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}       # 效果同上

        # 4.2 初始化英语词汇表大小计数器, 初始值为: 2, 因为已经包含: SOS和EOS
        english_word_n = 2

        # 5. 初始化法语词汇表 和 词汇表大小计数器
        french_word2index = {'SOS': 0, 'EOS': 1}
        french_word_n = 2

        # 6. (具体的)构建英语词汇表的动作.
        # 6.1 遍历所有的双语句子对, 获取英语句子中的单词.
        for pair in my_pairs:       # pair的数据格式: ['英语句子', '法语句子'], 例如: ['i m not doing it anymore .', 'je ne vais plus le faire .']
            # 6.2 对每个英语句子处理, 将句子按照 空格 切割成单词列表, 然后遍历, 获取到每个单词.
            for word in pair[0].split(' '):
                # 6.3 检查单词是否在词汇表中, 如果不存在, 就给改单词分配1个新索引, 新索引值 = 当前词汇表的大小.
                if word not in english_word2index:
                    english_word2index[word] = english_word_n
                    # 细节: 词汇表大小计数器 +1
                    english_word_n += 1

            # 6.4 构建法语词汇表
            for word in pair[1].split(' '):
                if word not in french_word2index:
                    french_word2index[word] = french_word_n
                    french_word_n += 1

        # 7. 构建反向映射表(索引到单词的映射)
        # 7.1 英语词汇表的反向映射.
        english_index2word = {v: k for k, v in english_word2index.items()}
        # 7.2 法语词汇表的反向映射.
        french_index2word = {v: k for k, v in french_word2index.items()}

        # 8. 打印词汇表统计信息.
        print(f'英语词汇表大小: {english_word_n}')     # 2803
        print(f'法语词汇表大小: {french_word_n}')      # 4345

        # 9. 返回构建的数据结构.
        # 返回: 英语词汇表映射, 英语单词反向映射, 英语词汇表大小, 法语词汇表映射, 法语单词反向映射, 法语词汇表大小, 双语句子对.
        return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs


# todo 4. 数据预处理 -> 构建数据集对象(DataSet)
# todo 4.1 调用 my_getdata()函数, 获取: 预处理好的数据结构.
english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()

# todo 4.2 定义 MyPairsDataset类 -> 表示: 数据集类.
class MyPairsDataset(Dataset):
    # 1. 初始化函数.
    def __init__(self, my_pairs):
        # 1.1 保存双语句子对.
        self.my_pairs = my_pairs
        # 1.2 计算样本总数, 样本数 = 双语句子对数.
        self.sample_len = len(my_pairs)

    # 2. 定义获取样本总数的方法.
    def __len__(self):
        return self.sample_len

    # 3. 定义获取单个样本的方法.
    def __getitem__(self, index):
        # 1. 修正索引值, 确保在有效范围内. 即: 索引不能小于0, 不能大于 样本总数 - 1
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按索引获取双语句子对, x: 英语句子, y: 法语句子.
        x = self.my_pairs[index][0]         # 'she s hot .'
        y = self.my_pairs[index][1]         # 'elle est chaude .'

        # 3. 英语句子文本转数值.
        # 3.1 按空格分割单词, 获取每个单词的索引.
        x = [english_word2index[word] for word in x.split(' ')]
        # 3.2 在句子末尾添加结束标记EOS
        x.append(EOS_token)
        # 3.3 将句子转换成张量, 并指定设备(CPU 或者 GPU)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)

        # 4. 法语句子文本转数值.
        y = [french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        # 5. 返回处理后的数据(英文张量, 法语张量)
        return tensor_x, tensor_y


# todo 5. 数据预处理 -> 定义函数, 获取DataLoader(数据加载器)对象.
def get_dataloader():
    # 1. 实例化数据集对象.
    my_dataset = MyPairsDataset(my_pairs)

    # 2. 创建数据加载器对象.
    # 参1: 数据集对象.  参2: 批次大小.  参3: 是否打乱数据(训练集打乱, 测试集不打乱).
    my_dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True)

    # 3. 测试数据加载器 -> 仅显示第1批数据.
    # for i, (x, y) in enumerate(my_dataloader):
        # 4. 打印英文句子相关信息.
        # print(f'x: {x.shape}, {x}')

        # 5. 打印法语句子相关信息.
        # print(f'y: {y.shape}, {y}')

        # 6. 测试结束 -> 只显示一批数据即可.
        # break

    # 7. 返回数据加载器对象.
    return my_dataloader


# todo 6.构建基于GRU的编码器.
"""
EncoderRNN类 实现思路分析:
    1. __init__函数, 定义 2 个层.
        self.embedding
        self.gru
    2. forward() 前向传播.
    3. 初始化 隐藏层输入数据. 
"""
class EncoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, input_size, hidden_size):
        """
        初始化属性信息的.
        :param input_size: 编码器词嵌入层的输入维度, 即: 词汇表的大小(2803个英语单词)
        :param hidden_size: 编码器的隐藏层维度, 即: 隐藏层单元的个数(256个隐藏单元)
        """
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数.
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 3. 实例化词嵌入层.
        # 输入: [batch_size, seq_len],    输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # 4. 实例化GRU层.
        # 参1: hidden_size: 输入的特征维度, 即: 词嵌入维度.
        # 参2: hidden_size: 隐藏层的维度, 即: 256
        # 参3: batch_first: 批次维数是否为第一个维度, 即(格式为): [批次大小, 句子长度, 词向量维度], 而不是默认的[句子长度, 批次大小, 词向量维度]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    # 2. 前向传播函数.
    def forward(self, input, hidden):
        """
        前向传播函数.
        :param input:  输入的单词索引序列, 即: [batch_size, seq_len] -> [1, 6]
        :param hidden: 初始的隐藏状态, 即: [num_layer, batch_size, hidden_size] -> [1, 1, 256]
        :return:
        """
        # 1. 通过词嵌入层, 将单词索引序列 转换为 单词向量序列.
        # 输入形状: [batch_size, seq_len] -> [1, 6]
        # 输出形状: [batch_size, seq_len, hidden_size]  -> [1, 6, 256]
        output = self.embedding(input)

        # 2. GRU层处理.
        # 输入:
        #   output: [batch_size, seq_len, hidden_size] >  [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        # 输出:
        #   output: [batch_size, seq_len, hidden_size] > [1, 6, 256]
        #   hidden: [num_layer, batch_size, hidden_size] > [1, 1, 256]
        output, hidden = self.gru(output, hidden)

        # 3. 返回GRU层的输出和最终的隐藏状态.
        return output, hidden

    # 3.初始化 隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 7. 测试基于GRU的编码器.
def dm_test_encoder():
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()

    # 2. 初始化编码器参数.
    vocab_size = english_word_n     # (英文)词汇表大小, 2803个英语单词
    hidden_size = 256               # 隐藏层维度, 256

    # 3. 创建编码器模型对象.
    my_encoder_gru = EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)

    # 4. (选做: CPU版可以不做) 将模型移动到指定的设备.
    my_encoder_gru = my_encoder_gru.to(device)

    # 5. 遍历数据集加载器, 进行测试.
    for i, (x, y) in enumerate(my_dataloader):
        # 5.1 打印输入英文句子索引张量 和 输入张量的形状.
        print(f'x: {x.shape}, {x}')     # x: torch.Size([1, 6]), tensor([14, 15, 43, 139, 888, 1])

        # 5.2 获取编码器的初始隐藏状态.
        h0 = my_encoder_gru.init_hidden()

        # 5.3 执行前向传播.
        output, hn = my_encoder_gru(x, h0)

        # 5.4 打印输出结果形状.
        print(f'output: {output.shape}')    # torch.Size([1, 6, 256])

        # 5.5 打印隐藏状态的形状.
        print(f'hidden: {hn.shape}')        # torch.Size([1, 1, 256])
        break       # 只看一组, 实际开发, 千万不要写.


# todo 8. 构建基于GRU的解码器 -> 版本1: 无Attention(注意力机制)
class DecoderRNN(nn.Module):
    # 1. 初始化函数.
    def __init__(self, output_size, hidden_size):
        """
        初始化属性信息
        :param output_size: 解码器输出维度, 即: 目标语言(法语)词汇表大小
        :param hidden_size: 解码器隐藏层维度, 即: 每个词向量的特征数(256)
        """
        # 1. 初始化父类信息
        super().__init__()
        # 2. 保存输入参数.
        self.output_size = output_size
        self.hidden_size = hidden_size
        # 3. 创建词嵌入层, 输入: [batch_size, seq_len], 输出: [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 4. 创建GRU层.
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # 5. 创建线性层(输出层)
        self.out = nn.Linear(hidden_size, output_size)
        # 6(了解). 创建softmax层.
        self.softmax = nn.LogSoftmax(dim=-1)



    # 2. 前向传播函数.
    def forward(self, input, hidden):
        # 1. 词嵌入处理.
        output = self.embedding(input)
        # 2. ReLU激活函数处理.
        output = F.relu(output)
        # 3. GRU层处理 -> 大白话处理: 都是[1, 1, 256]
        # 输入时: output -> [batch_size, seq_len, hidden_size],   hidden -> [num_layers, batch_size, hidden_size]
        # 输出时: output -> 形状同上,   hidden -> 形状同上
        output, hidden = self.gru(output, hidden)

        # 4. 线性层 和 softmax层处理.
        output = self.softmax(self.out(output[0]))

        # 5. 返回结果.
        return output, hidden


    # 3. 初始化隐藏层输入数据.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# todo 9. 测试基于GRU的解码器 -> 测试版本1: 无Attention(注意力机制)
def dm_test_decoder():
    # 1. 获取数据集加载器对象.
    my_dataloader = get_dataloader()
    # 2. 初始化编码器模型, 并移动到GPU.
    my_encoder_gru = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device)
    print(f'my_encoder_gru: {my_encoder_gru}')
    
    # 3. 初始化解码器模型, 并移动到GPU.
    my_decoder_gru = DecoderRNN(output_size=french_word_n, hidden_size=256).to(device)
    print(f'my_decoder_gru: {my_decoder_gru}')

    # 4. 完整的编码 -> 解码流程测试.
    # 4.1 从数据集加载器中获取1个批次的样本(即: 1条数据)
    for i, (x, y) in enumerate(my_dataloader):
        # 4.2 打印输入数据的信息.
        # print(f'输入数据信息(英语句子): {x.shape}, {x}')  # 输入数据信息(英语句子): torch.Size([1, 6]), tensor([[ 77,  78, 147,  24,   4,   1]])
        # print(f'输入数据信息(法语句子): {y.shape}, {y}')  # 输入数据信息(法语句子): torch.Size([1, 7]), tensor([[123, 297, 126, 246, 384,   5,   1]])
        print(f'输入数据信息(英语句子): {x.shape}')
        print(f'输入数据信息(法语句子): {y.shape}')

        # 4.3 编码过程: 将英文句子编码为 隐藏状态.
        # 4.3.1 初始化编码器.
        h0 = my_encoder_gru.init_hidden()
        # 4.3.2 (编码器)前向传播.
        encoder_output_c, hidden = my_encoder_gru(x, h0)
        # print(f'编码器输出: {encoder_output_c.shape}')       # 编码器输出: torch.Size([1, 6, 256])

        # 4.4 解码过程: 将隐藏状态解码为法语句子.
        # print(f'观察: 最后1个时间步的output输出: {encoder_output_c[0][-1].shape}') # [6, 256] -> [256]

        # 4.4.1 具体的解码过程 -> 逐个字符生成, 将隐藏状态解码为法语句子.
        # 遍历目标句子(法语句子)的每个时间步.
        for i in range(y.shape[1]):
            # 4.4.2 提取当前时间步的 目标词索引.
            # y[0][i]:     取出batch中第1个样本的第i个词的索引.
            # view(1, -1): 将标量转为[1, 1]的形状, 匹配: 解码器输入要求.
            tmp = y[0][i].view(1, -1)
            # 4.4.3 解码器的前向传播.
            output, hidden = my_decoder_gru(tmp, hidden)
            # 4.4.4 打印解码器的输出信息.
            # print(f'output每个时间步解析出来的概率分布: {output.size()}, {output.shape}')
            print(f'第 {i + 1} 个法语单词的预测概率分布: {output.shape}, {output.shape}')

        print('\n' * 5)


# todo 10. 构建基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
class AttnDecoderRNN(nn.Module):
    # todo 10.1 初始化函数.
    # 参1: 目标(法语)词汇表大小, 参2: 隐藏层维度(和编码器一致), 参3: 随机失活概率, 参4: (句子)最大长度
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 保存输入参数
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 3. 创建词嵌入层.
        # 输入形状: [batch_size, seq_len] -> [1, 1]
        # 输出形状: [batch_size, seq_len, hidden_size] -> [1, 1, 256]
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # 4. 注意力权重计算层: 计算查询向量 和 编码器输出的匹配程度.
        # 参1: 拼接后的查询向量 和 隐藏状态 -> [1, 1, 512]
        # 参2: 注意力的权重分布 -> [1, 1, 10]  (最大)10个词
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 5. 注意力融合层, 将 词嵌入 和 注意力权重进行融合.
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # 6. dropout层, 随机丢弃部分神经元, 防止过拟合.
        self.dropout = nn.Dropout(self.dropout_p)

        # 7. 创建GRU层: 处理序列数据, 维持隐藏状态.
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 9. 对数softmax()层, 将输出映射为 对数softmax()概率分布
        self.softmax = nn.LogSoftmax(dim=-1)


    # todo 10.2 前向传播函数.
    # 参1: input 当前时间步的输入词索引 -> [batch_size, 1]
    # 参2: hidden 上一个时间步的隐藏状态 -> [1, batch_size, hidden_size]
    # 参3: encoder_outputs 编码器所有时间步的输出 -> [batch_size, seq_len, hidden_size]
    def forward(self, input, hidden, encoder_outputs):
        # 1. 词嵌入层, 输入形状: [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        # [1, 1] -> [1, 1, 256]
        embedded = self.embedding(input)

        # 2. 应用Dropout(随机失活层), 随机失活.
        embedded = self.dropout(embedded)

        # 3. 计算注意力权重.
        # step1: torch.cat((embedded[0], hidden[0]), 1))                ->  [1, 512]
        # step2: self.attn(torch.cat((embedded[0], hidden[0]), 1)       ->  [1, 10]
        # step3: 应用softmax()层, 映射为: [1, 10] -> [1, 10]
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 4. 计算注意力上下文.
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        # 5.注意力融合层.
        output = torch.cat((embedded[0], attn_applied[0]), 1)       # [1, 1, 512]
        output = self.attn_combine(output).unsqueeze(0)                          # [1, 1, 256]
        # 6. 激活函数处理.
        output = F.relu(output)
        # 7. GRU层: 处理序列数据, 维持隐藏状态.
        output, hidden = self.gru(output, hidden)       # [1, 1, 256]
        # 8. 输出层: 将GRU的隐藏状态映射为: (法语)目标词汇表大小.
        output = self.softmax(self.out(output[0]))      # [1, 4345]

        # 9. 返回结果.
        # 参1: 当前时间步的输出概率分布          ->  [1, 4345]
        # 参2: 更新后的隐藏状态(本次的隐藏状态)   ->  [1, 1, 256]
        # 参3: 注意力权重分布(用于可视化分析, 如果不做, 可以不返回)     -> [1, 10]
        return output, hidden, attn_weights


    # todo 10.3 定义初始化的隐藏状态.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



# todo 11. 测试基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
def dm_test_attn_decoder():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()

    # 2. 模型初始化阶段
    # 2.1 创建编码器对象.
    # 参1: 英语词汇表大小(2803), 参2: 隐藏层维度
    my_encoder = EncoderRNN(english_word_n, 256).to(device)

    # 2.2 创建解码器对象.
    # 参1: 法语词汇表大小(4345), 参2: 隐藏层维度
    my_decoder = AttnDecoderRNN(french_word_n, 256).to(device)

    # 3. 模型训练(推理)阶段
    # 3.1 从数据加载器中获取1个样本.
    for i, (x, y) in enumerate(my_dataloader):
        # 3.2 打印输入信息
        print(f'x(英语句子): {x.shape}, {x}')       # torch.Size([1, 6]),
        print(f'y(法语句子): {y.shape}, {y}')       # torch.Size([1, 8]),

        # 3.3 编码过程 -> 将英语句子 编码成 隐藏状态 -> (中间语义张量C)
        hidden = my_encoder.init_hidden()
        # output: 所有时间步的隐藏状态: [1, seq_len, 256]
        # hidden: 最后一个时间步的隐藏状态: [1, 1, 256]
        output, hidden = my_encoder(x, hidden)

        # 3.4 准备编码器的输出 -> 用于Attention机制.
        # 创建1个固定大小的张量, 用于存储编码器输出, 形状为: [10, 256]
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)

        # 3.5 将编码器实际输出 复制到 固定大小的张量中.
        for idx in range(output.shape[1]):
             encoder_output_c[idx] = output[0, idx]

        # 3.6 解码过程: 将隐藏状态 解码为 法语句子(必须逐词翻译)
        # 3.6.1 遍历目标句子的每个时间步.
        for i in range(y.shape[1]):
            # 3.6.2 提取当前时间步的目标词索引.
            tmp = y[0][i].view(1, -1)       # 例如: [[1595]]

            # 3.6.3 执行解码器的前向传播.
            # (实际)参数列表解释
            # tmp: 当前时间步的输入词索引, 形状: [1, 1]
            # hidden: 上一个时间步的隐藏状态, 形状: [1, 1, 256]
            # encoder_output_c: 编码器(所有时间步)的输出, 形状: [10, 256]

            # 返回值参数列表解释:
            # output: 当前时间步的输出概率分布, 形状: [1, 4345]
            # hidden: 当前时间步的隐藏状态, 形状: [1, 1, 256]
            # attn_weights: 当前时间步的注意力权重分布, 形状: [1, 10]

            #                                          Q    K         V
            output, hidden, attn_weights = my_decoder(tmp, hidden, encoder_output_c)

            # 3.6.4 打印结果.
            print(f'解码output.shape: {output.shape}')    # [1, 4345]
            print(f'解码hidden.shape: {hidden.shape}')    # [1, 1, 256]
            print(f'解码attn_weights.shape: {attn_weights.shape}')    # [1, 10]
            print('\n' * 3)

        # 只看1个句子(样本)
        break


# todo 12. 构建模型内部迭代训练函数 -> 即: 完成单批次的训练过程.
# todo 12.1 定义模型训练参数.
# 学习率, 训练轮数, Teacher_Forcing比例, 输出信息打印间隔(每训练1000条打印一次),  绘图间隔(每训练100条绘图一次)
my_lr, epochs, teacher_forcing_ratio, print_interval_num, plot_interval_num = 1e-4, 5, 0.5, 1000, 100

# todo 12.2 定义函数, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
def train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss):
    """
    函数作用, 实现: 单批次训练, 完成1个样本的 编码 -> 解码 -> 反向传播 -> 优化参数...
    :param x: 输入序列, 即: 英语句子, 形状为: [batch_size = 1, seq_len]
    :param y: 目标序列, 即: 法语句子, 形状为: [batch_size = 1, seq_len]
    :param my_encoder_rnn: 编码器对象
    :param my_attn_decoder_rnn: 解码器对象(带注意力机制)
    :param myadam_encode: 编码器优化器
    :param myadam_decode: 解码器优化器
    :param my_crossentropy_loss: 损失函数
    :return:
    """
    # 1. 编码阶段, 将输入的序列转换为上下文向量, 初始的隐藏状态: [1, 1, 256]
    encoder_hidden = my_encoder_rnn.init_hidden()
    # 编码器的前向传播.
    encoder_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

    # 2. 解码参数准备
    # 2.1 构建编码器的输出张量, 用于: 注意力计算, 形状为: [10, 256]
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
    # 2.2 复刻实际的编码器输出 -> 固定长度张量, 即: [假设6个单词, 256] -> [10, 256]
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]

    # 2.3 解码器初始化隐藏状态.
    decoder_hidden = encoder_hidden     # [1, 1, 256]

    # 2.4 解码器的初始输入.
    input_y = torch.tensor([[SOS_token]], device=device)        # [1, 1]

    # 3. 初始化损失值
    my_loss = 0.0
    y_len = y.shape[1]      # 目标序列长度(要预测的法语句子长度), 例如: 9

    # 4. 根据概率值决定是否用 Teacher Forcing.
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # 4.1 走这里, 说明使用 Teacher Forcing, 就: 用真实标签作为下一步的输入.
        for i in range(y_len):
            # 4.1.1 解码器的前向传播
            # 输入: input_y -> [1, 1], decoder_hidden -> [1, 1, 256], encoder_output_c -> [10, 256]
            # 输出: output_y -> [1, 4345], decoder_hidden -> [1, 1, 256], attn_weights -> [1, 10]
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.1.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.1.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.1.4 下个时间步的输入 直接使用 真实标签.
            input_y = y[0][i].view(1, -1)   # [1, 1]
    else:
        # 4.2 非Teacher Forcing, 就: 用上一时间步的预测结果作为下一步的输入.
        for i in range(y_len):
            # 4.2.1 解码器前向传播
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(input_y, decoder_hidden, encoder_output_c)
            # 4.2.2 获取当前时间步的真实标签.
            target_y = y[0][i].view(1)
            # 4.2.3 累加损失
            my_loss += my_crossentropy_loss(output_y, target_y)
            # 4.2.4 获取预测的下一个词, 即: 获取概率最高的词的索引和概率.
            topv, topi = output_y.topk(1)
            # 4.2.5 如果预测到句子的结束标记, 则停止预测.
            if topi.squeeze().item() == EOS_token:
                break
            # 4.2.6 走到这里, 说明没有预测到结束标记, 则: 将预测的词 作为 下一步的输入.
            input_y = topi.detach()     # [1, 1]

    # 5. 反向传播和参数更新.
    myadam_encode.zero_grad()       # 梯度清零, 编码器.
    myadam_decode.zero_grad()       # 梯度清零, 解码器.

    my_loss.backward()              # 反向传播.
    myadam_encode.step()            # 参数更新, 编码器.
    myadam_decode.step()            # 参数更新, 解码器.

    # 6. 返回平均损失.
    return my_loss.item() / y_len


# todo 13. 构建模型训练函数 -> 即: 完成所有批次的训练过程, 即: 多轮, 多批次训练过程.
def train_seq2seq():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 模型初始化, 记得把模型移动到GPU上, 我的电脑(训练1轮): CPU训练时间 50分钟, GPU训练时间: 20分钟
    # 2.1 编码器, 输入维度 = 英文词汇表大小2803, 隐藏层维度: 256
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    # 2.2 解码器, 输入维度 = 法语词汇表大小4345, 隐藏层维度: 256
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256, 0.1, 10).to(device)

    # 3. 优化器初始化, 使用: Adam优化器, 学习率: 1e-4
    myadam_encode = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)       # 编码器优化器.
    myadam_decode = optim.Adam(my_attn_decoder_rnn.parameters(), lr=my_lr)  # 解码器优化器.

    # 4. 损失函数初始化, 使用: NLLLoss
    my_crossentropy_loss = nn.NLLLoss()

    # 5. 训练参数初始化.
    plot_loss_list = []         # 存储绘图用的损失值.

    # 6. 具体的多轮, 多批次训练过程.
    # 6.1 外层循环, 控制训练轮数.
    for epoch_idx in range(1, epochs + 1):
        # 6.1.1 初始化本轮的损失累加器
        print_loss_total, plot_loss_total = 0.0, 0.0
        # 6.1.2 记录本轮开始训练时间.
        start_time = time.time()

        # 6.2 内层循环, 遍历数据集的每个样本(即: 每轮具体的 所有批次训练过程)
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            # 6.2.1 调用内部训练函数, 完成: 单批次(单样本)的训练过程.
            myloss = train_iters(x, y, my_encoder_rnn, my_attn_decoder_rnn, myadam_encode, myadam_decode, my_crossentropy_loss)

            # 6.2.2 累加损失.
            print_loss_total += myloss
            plot_loss_total += myloss

            # 6.3 打印训练日志(每 print_interval_num=1000 个样本打印一次)
            if item % print_interval_num == 0:
                # 计算平均损失.
                print_loss_avg = print_loss_total / print_interval_num
                # 重置损失累加器.
                print_loss_total = 0.0
                # 打印训练信息: 轮次, 平均损失, 耗时.
                print(f'轮次: {epoch_idx}, 平均损失: {print_loss_avg:.4f}, 耗时: {time.time() - start_time:.4f} s(秒)!')

            # 6.4 记录损失用于绘图(每 plot_interval_num=100 个样本记录一次)
            if item % plot_interval_num == 0:
                # 计算平均损失.
                plot_loss_avg = plot_loss_total / plot_interval_num
                # 存储损失值.
                plot_loss_list.append(plot_loss_avg)
                # 重置损失累加器.
                plot_loss_total = 0.0

            # 扩展: 每轮训练3000个样本后, 结束训练, 实际开发, 这个代码万万不能写.
            if item > 3000:
                break

        # 6.3 走到这里, 说明一轮训练完毕, 保存模型.
        torch.save(my_encoder_rnn.state_dict(), f'./model/my_encoder_rnn_{epoch_idx}.pth')  # pickle文件后缀, .pth, .pkl, .pickle
        torch.save(my_attn_decoder_rnn.state_dict(), f'./model/my_attn_decoder_rnn_{epoch_idx}.pth')

    # 7. 训练结束后, 绘制损失曲线.
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/seq2seq_loss.png')
    plt.show()

    # 8. 训练结束, 返回结果.
    return plot_loss_list           # 训练损失列表(每训练100个样本的平均损失)


# todo 14. 构建模型评估 -> 用训练好的seq2seq模型进行 翻译.
def evaluate_seq2seq(x, my_encoder_rnn, my_attn_decoder_rnn):   # 英语句子, 编码器, 解码器(带注意力机制)
    # 0. 关闭梯度计算, 节省内存并加速推理 -> 只适用于 模型的预测过程.
    with torch.no_grad():
        # 1.编码阶段: 将输入的英文句子 -> 隐藏状态.
        encode_hidden = my_encoder_rnn.init_hidden()
        # 本次的输出, 本次的隐藏状态 = 编码器模型(本次的输入, 上一时刻的隐藏状态)
        encode_output, encode_hidden = my_encoder_rnn(x, encode_hidden)

        # 2. 解码器参数准备.
        # 2.1 构建固定长度的编码器输出张量.
        encode_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encode_output_c[idx] = encode_output[0, idx]

        # 2.2 解码器的隐藏状态.
        decode_hidden = encode_hidden

        # 2.3 解码器的初始输入, 句子的开始标记.
        input_y = torch.tensor([[SOS_token]], device=device)

        # 3. 自回归解码过程(逐个生成目标句子)
        # 定义遍历, 记录: 存储解码后的法语单词
        decode_words = []
        # 初始化注意力矩阵
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)
        # 遍历, 开始解码.
        for idx in range(MAX_LENGTH):
            # 3.1 解码器, 前向传播.
            # 输入: 当前输入词的索引(Q), 解码隐藏状态(K), 编码器输出张量(V))
            # 输出: 下个词的概率分布, 更新后的隐藏状态, 注意力权重分布矩阵
            output_y, decode_hidden, attn_weights = my_attn_decoder_rnn(input_y, decode_hidden, encode_output_c)

            # 3.2 记录注意力矩阵.
            decoder_attentions[idx] = attn_weights
            # 3.3 预测下个词.
            topv, topi = output_y.topk(1)
            # 3.4 处理终止条件, 如果预测到EOS标记, 结束生成.
            if topi.squeeze().item() == EOS_token:
                break
            else:
                # 3.5 否则, 则添加预测的词到结果列表.
                decode_words.append(french_index2word[topi.squeeze().item()])
            # 3.6 更新输入: 把当前预测词作为下个时间步的输入.
            input_y = topi.detach()

        # 4. 返回解码结果 和 注意力矩阵.
        return decode_words, decoder_attentions[:idx + 1]


# todo 15. 模型评估函数调用, 记载模型, 对比 自定义样本进行 翻译.
# 模型路径
PATH1 = './model/my_encoder_rnn_5.pth'
PATH2 = './model/my_attn_decoder_rnn_5.pth'

# 定义函数
def dm_test_seq2seq_evaluate():
    # 1. 获取数据加载器对象
    my_dataloader = get_dataloader()
    # 2. 加载编码器模型.
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)
    # map_location: 确保在CPU和GPU(Cuda)都能加载. 正常应该是用GPU训练, 就用GPU预测, 用CPU训练, 就用CPU预测.
    # 写了 map_location能实现: 用GPU训练, 用CPU预测.
    # weights_only: 只加载模型权重参数.
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), False)
    print(f'my_encoder_rnn编码器模型架构: {my_encoder_rnn}')

    # 3. 加载解码器模型
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), False)
    print(f'my_attn_decoder_rnn解码器模型架构: {my_attn_decoder_rnn}')

    # 4. 自定义测试样本.
    my_sample_pairs = [
        # 格式: ['英文句子', '法语句子']
        ['i m the spokesperson for this organization .', 'je suis le porte parole de cette institution .'],
        ['i am thinking about buying a new parasol .', 'je songe a acheter un nouveau parasol .'],
        ['he is able to swim very fast .', 'il est capable de nager tres vite .']
    ]
    print(f'自定义测试样本: {my_sample_pairs}')

    # 5. 对每个样本进行翻译.
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]     # 英语句子
        y = pair[1]     # 法语句子

        # 5.1 文本数值化, 英语句子 -> 索引序列
        tmpx = [english_word2index[word] for word in x.split(' ')]
        tmpx.append(EOS_token)      # 添加结束标记.
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 5.2 模型预测.
        decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)

        # 5.3 把翻译结果转成句子格式.
        output_sentence = ' '.join(decode_words)
        print(f'输入(原始英文句子): {x}')
        print(f'输入(原始法语句子): {y}')
        print(f'输出(翻译后的法语句子): {output_sentence}')
        print(' -.- ' * 10, '\n')


# todo 16. 绘制注意力图的函数 -> Attention张量绘图(看看即可, 无需编写)
def dm_test_attention():
    # 1. 获取数据加载器对象.
    my_dataloader = get_dataloader()
    # 2. 实例化模型.
    # 2.1 编码器模型
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), False)

    # 2.2 解码器模型.
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), False)

    # 3. 定义遍历, 记录: 英语句子
    sentence = 'we re both teachers .'
    # 对上述的样本进行数值化.
    tmpx = [english_word2index[word] for word in sentence.split(' ')]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    # 4. 模型预测.
    decode_words, attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder_rnn)
    print(f'decode_words: {decode_words}')


    # 5. 绘制注意力图.
    plt.matshow(attentions.numpy())     # 以矩阵列表的形式 显示.  matrix: 矩阵.
    # 保存图像.
    plt.savefig('./img/s2s_attn.png')
    plt.show()

    # 6. 打印下参数.
    print(f'attentions.numpy(): {attentions.numpy()}')
    print(f'attentions.size(): {attentions.size()}')



# todo n.测试代码
if __name__ == '__main__':
    # 1. 测试用例: 包括多种语言和特殊符号的字符串.
    # s = ' I Love You! .?! 我#爱你 end'
    # print(f'(原始的字符串)s: {s}')
    # print(' -.- ' * 10)
    # normalizeString(s)

    # 2. 测试数据预处理函数.
    # english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs = my_getdata()
    # print(f'英语词汇表映射: {english_word2index}')
    # print(f'英语单词反向映射: {english_index2word}')
    # print(f'英语词汇表大小: {english_word_n}')
    # print(f'法语词汇表映射: {french_word2index}')
    # print(f'法语单词反向映射: {french_index2word}')
    # print(f'法语词汇表大小: {french_word_n}')
    # print(f'双语句子对: {my_pairs[:5]}')

    # 3. 测试数据加载器.
    # get_dataloader()

    # 4. 测试基于GRU的编码器.
    # dm_test_encoder()

    # 5. 测试基于GRU的解码器 -> 测试版本1: 无Attention(注意力机制)
    # dm_test_decoder()

    # 6. 测试基于GRU的解码器 -> 测试版本2: 带Attention(注意力机制)
    # dm_test_attn_decoder()

    # 7. 模型训练.
    # train_seq2seq()

    # 8. 模型评估.
    # dm_test_seq2seq_evaluate()

    # 9. 绘制注意力图.
    dm_test_attention()
