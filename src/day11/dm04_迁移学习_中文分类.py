"""
案例:
    NLP任务 -> 中文分类 案例.
任务介绍:
    直接加载预训练模型进行输入文本的特征表示, 后接自定义网络进行微调输出结果.
数据介绍:
    数据文件有3个, 分别是: train.csv, test.csv, validation.csv 三个文件的数据格式都是一样的.
        label                   text
        标签(1:好评,0:差评)       文本
"""

# 导包
import torch                                    # Pytorch深度学习框架, 提供张量计算 和 自动微分功能.
import torch.nn as nn                           # neural network, 神经网络, 例如: 线性层, 卷积层...
from torch.utils.data import DataLoader         # 数据加载器, 支持: 批量处理数据
from datasets import load_dataset               # HuggingFace的数据集加载工具, 可以加载: 本地文件, 或者公开数据源(GLUE, SQuAD)...
import time                                     # 时间模块
from tqdm import tqdm                           # 导入进度条库
from transformers import BertTokenizer, BertModel   # 导入BERT相关组件(中文分词器, 预训练的BERT模型)
from transformers import AdamW                  # 导入AdaW优化器, 内置权重衰减 -> 防止过拟合...
from rich import print                          # 终端打印美化.

# 使用GPU或者MPS(苹果的M系列芯片)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# todo 1. 加载分词器.
my_tokenizer = BertTokenizer.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\bert-base-chinese')
# todo 2. 加载模型.
my_pre_model = BertModel.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\bert-base-chinese').to(device)


# todo 3.定义函数, 加载数据集. 即: 从CSV文件加载 训练集, 测试集, 验证集数据.
def dm_file2dataset():
    # 前言: 加载数据有两种方式, 我用 训练集给大家演示一下, 其它都一样.
    # ----------------------------------- 加载训练数据集 -----------------------------------
    # 思路1: path 设定为 目标文件的 文件夹路径.
    # 1. load_dataset()方式, 加载csv格式的训练数据.
    # train_dataset = load_dataset(path='./data', data_files='train.csv', split='train')

    # 思路2: path 设定为 文件类型.
    train_dataset = load_dataset(path='csv', data_files=r'./data/train.csv', split='train')

    # 2. 打印训练集的基本信息.
    print(f'train_dataset: {train_dataset}')
    print(f'训练数据的长度: {len(train_dataset)}')  #  9600
    print(f'训练数据(第1条): {train_dataset[0]}')
    print(f'训练数据(前3条): {train_dataset[:3]}')
    print(' -.- ' * 10)

    # ----------------------------------- 加载测试数据集 -----------------------------------
    test_dataset = load_dataset(path='csv', data_files=r'./data/test.csv', split='train')
    print(f'test_dataset: {test_dataset}')
    print(f'测试数据集的长度: {len(test_dataset)}')     # 1200
    print(f'测试数据(前3条): {test_dataset[:3]}')


    # 加载验证集数据集.
    valid_dataset = load_dataset(path='csv', data_files=r'./data/validation.csv', split='train')
    print(f'valid_dataset: {valid_dataset}')
    print(f'验证数据集的长度: {len(valid_dataset)}')    # 1200
    print(f'验证数据(前3条): {valid_dataset[:3]}')


    # 3. 返回结果(数据集对象)
    return train_dataset, test_dataset, valid_dataset


# todo 4. 数据整理函数, 处理批次数据, 统一格式并转换为: 模型输入.
def collate_fn1(data):      # data = 8条数据/批次
    # 1. 提取文本 和 标签.
    sents = [item["text"] for item in data]     # 文本内容
    labels = [item["label"] for item in data]   # 标签

    # 2. 批量编码文本: 将文本转成模型输入格式 -> 通过分词器实现.
    inputs = my_tokenizer.batch_encode_plus(
        sents,                  # 输入文本列表(待处理的文本列表)
        truncation=True,        # 启用文本截断
        max_length=300,         # 最大序列长度
        padding='max_length',   # 启用填充
        return_tensors='pt',    # 返回二维张量
        return_length=True      # 返回编码后的序列长度(可选)
    )

    # 3. 提取编码结果.
    print(f'inputs: {inputs}')      # 就是咱们以前处理的 input_ids, token_type_ids, attention_mask 这三个组成的字典.
    # 3.1 模型输入 token_id, 形状: [batch_size, max_length]
    input_ids = inputs['input_ids']
    # 3.2 token类型id, 形状: [batch_size, max_length], 单句为0, 句子对为1
    token_type_ids = inputs['token_type_ids']
    # 3.3 注意力掩码(1: 真实, 0: 填充)
    attention_mask = inputs['attention_mask']

    # 4. 将上述的标签 -> 转成张量.
    labels = torch.LongTensor(labels)

    # 5. 返回模型所需的输入张量.
    return input_ids, token_type_ids, attention_mask, labels


# todo 5. 获取数据加载器.
def get_dataloader():
    # 1. 加载训练数据集.
    train_dataset = load_dataset(path='csv', data_files=r'./data/train.csv', split='train')
    # 2. 将上述的数据集对象, 封装成数据加载器.
    my_dataloader = DataLoader(
        dataset=train_dataset,          # 输入的数据集对象
        batch_size=8,                   # 批次大小(即: 每批包含的样本数)
        shuffle=True,                   # 训练数据集是否打乱.
        drop_last=True,                 # 训练数据集是否删除最后1个批次(批次数不足1个批次的数据)
        collate_fn=collate_fn1,         # 指定数据整理函数, 即: 每批次的数据都要被这个函数处理.
    )

    # 3. 返回数据加载器.
    return my_dataloader




# todo n.测试代码.
if __name__ == '__main__':
    # 1.演示: 加载数据集.
    # dm_file2dataset()

    # 2. 创建数据加载器, 并测试.
    train_dataloader = get_dataloader()
    # 获取一批次数据, 看看效果.
    for batch in train_dataloader:
        print(batch)
        break       # 只看1批数据即可.