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
import warnings
warnings.filterwarnings("ignore")       # 忽略警告


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
import torch.optim as optim                     # 优化器, 例如: SGD, Adam, AdaW...

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
    # print(f'inputs: {inputs}')      # 就是咱们以前处理的 input_ids, token_type_ids, attention_mask 这三个组成的字典.
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


# todo 6. 自定义下游模型, 基于BERT的文本分类模型.
class AiModel(nn.Module):
    # 1. 初始化函数
    def __init__(self):
        # 1. 初始化父类成员.
        super().__init__()
        # 2. 定义全连接层, 目的: 把bert模型的768维 -> 2维(2分类)
        self.linear = nn.Linear(768, 2)


    # 2. 前向传播函数.
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        前向传播函数, 处理数据,  获取最终的2分类结果.
        :param input_ids: 文本的数字编码, 形状: [batch_size, max_len]
        :param token_type_ids: 句子类型标记, 形状: [batch_size, max_len]
        :param attention_mask: 注意力掩码, 形状: [batch_size, max_len]
        :return:
        """
        # 1. 不计算BERT预训练模型的梯度(冻结参数)
        # 作用: 避免更新预训练模型参数, 仅训练自定义模型(分类层)的参数.
        # with torch.no_grad():
        #     # 2. 获取BERT模型的输出.
        #     bert_output = my_pre_model(
        #         input_ids=input_ids,
        #         token_type_ids=token_type_ids,
        #         attention_mask=attention_mask,
        #     )

        # 2. 获取BERT模型的输出.
        bert_output = my_pre_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # 3. 使用bert的 pooler_output 进行分类.
        # bert_output.pooler_output: [batch_size, 768]
        output = self.linear(bert_output.pooler_output)

        # 4. 返回处理后的分类结果
        return output


# todo 7. 测试模型结构函数.
def use_AiModel():
    # 1. 获取数据加载器.
    my_dataloader = get_dataloader()
    # 2. 实例化模型 -> 创建基于BERT的分类模型.
    # 3. 将模型移动到GPU中.
    my_model = AiModel().to(device)
    # 4. 迭代数据并测试输出, 细节: 这里我们仅仅测试1批次的数据, 避免长时间运行.
    for input_ids, token_type_ids, attention_mask, labels in my_dataloader:
        # 4.1 把上述的数据移动到GPU中.
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 4.2 前向传播, 计算结果.
        output = my_model(input_ids, token_type_ids, attention_mask)
        print(f'output: {output.shape}, {output}')      # 形状: [batch_size, 2]
        break       # 我们仅仅测试1批次的数据, 避免长时间运行.


# todo 8. 训练模型. 流程: 加载数据 -> 初始化模型 -> 配置训练参数 -> 迭代训练 -> 保存模型.
def train_model():
    # 1. 加载训练集
    train_dataset = load_dataset(path='csv', data_files=r'./data/train.csv', split='train')
    # 2. 创建模型, 并移动到指定设备.
    my_model = AiModel().to(device)
    # 3. 冻结BERT预训练模型的参数.
    for param in my_pre_model.parameters():
        # param.requires_grad_ = False # 效果一样
        param.requires_grad_(False)

    # 4. 创建损失函数对象.
    criterion = nn.CrossEntropyLoss(reduction='mean')
    # 5. 创建优化器对象.
    optimizer = optim.Adam(my_model.parameters(), lr=1e-3)
    # 6. 设置模型为训练模式.
    my_model.train()
    # 7. 开始具体的训练过程 -> 假设训练3轮.
    for epoch_idx in range(3):          # epoch_idx的值: 0, 1, 2
        # 7.1 获取本轮的开始时间.
        start_time = time.time()
        # 7.2 获取数据加载器
        my_dataloader = get_dataloader()

        # 7.3 具体的 本轮的 每批次的 训练过程.
        for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(my_dataloader), start=1):
            # 7.3.1 将参数移动到指定设备(GPU)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # 7.3.2 模型前向传播, 获取模型输出结果.
            output = my_model(input_ids, token_type_ids, attention_mask)
            # 7.3.3 计算损失.
            loss = criterion(output, labels)    # 预测值, 真实值.
            # 7.3.4 反向传播(梯度清零 + 反向传播 + 参数更新)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 7.3.5 输出训练进度 -> 20批次打印一下.
            if i % 20 == 0:
                # 获取预测结果, 对输出获取最大值索引, 即: 概率.
                tem = torch.argmax(output, dim=-1)
                # 计算准确率
                acc = (tem == labels).sum().item() / len(labels)
                # 计算使用时间
                use_time = time.time() - start_time
                # 打印训练信息
                print(f'轮数: {epoch_idx + 1}, 迭代的步数: {i}, 当前的损失值: {loss:.2f}, (Acc)准确率: {acc:.2f}, 耗时: {use_time:.2f}s')

        # 7.4 走到这里, 说明本轮训练完毕, 保存模型.
        torch.save(my_model.state_dict(), f'./model/classification_{epoch_idx + 1}.bin')


# todo 9. 模型评估函数. 流程: -> 加载数据 -> 创建加载器 -> 加载模型 -> 批量预测 -> 计算准确率 -> 打印结果.
def evaluate_model():
    # 1. 加载(测试)数据集.
    test_dataset = load_dataset(path='csv', data_files=r'./data/test.csv', split='train')
    # 2. 创建加载器对象.
    my_dataloader = DataLoader(
        dataset=test_dataset,           # 输入的数据集对象
        batch_size=8,                   # 批次大小(即: 每批包含的样本数)
        shuffle=False,                   # 训练数据集是否打乱.
        drop_last=True,                 # 训练数据集是否删除最后1个批次(批次数不足1个批次的数据)
        collate_fn=collate_fn1,         # 指定数据整理函数, 即: 每批次的数据都要被这个函数处理.
    )
    # 3. 加载训练好的模型.
    path = './model/classification_3.bin'
    my_model = AiModel().to(device)
    my_model.load_state_dict(torch.load(path))  # 加载训练好的模型参数.

    # 4. 初始化评估参数.
    correct, total = 0, 0       # 预测正确的样本数, 预测的总样本数.
    # 5. 设置模型为评估模式.
    my_model.eval()

    # 6. 迭代预测, 通过数据加载器获取每批次数据, 预测, 统计即可.
    for i, (input_ids, token_type_ids, attention_mask, labels) in enumerate(tqdm(my_dataloader), start=1):
        # 6.1 将参数移动到指定设备(GPU)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        # 6.2 不计算梯度.
        with torch.no_grad():
            # 6.3 前向传播 -> 输入数据, 给模型 -> 获取结果.
            output = my_model(input_ids, token_type_ids, attention_mask)
            # 6.4 获取预测结果, 对输出获取最大值索引, 即: 概率.
            temp = torch.argmax(output, dim=-1)
            # 6.5 统计预测正确的样本数.
            correct += (temp == labels).sum().item()
            # 6.6 统计总样本数.
            total += len(labels)
            # 6.7 输出预测进度, 间隔20步打印一次.
            if i % 20 == 0:
                # 打印平均准确率
                print(f'Acc(准确率): {correct / total:.2f}')

                # 解码第1个样本的文本, 并打印预测结果.
                text_list = my_tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                print(f'原始文本是: {text_list}')
                print(f'预测的标签是: {temp[0]}, 真实标签是: {labels[0]}')




# todo n.测试代码.
if __name__ == '__main__':
    # 1.演示: 加载数据集.
    # dm_file2dataset()

    # # 2. 创建数据加载器, 并测试.
    # train_dataloader = get_dataloader()
    # # 获取一批次数据, 看看效果.
    # for batch in train_dataloader:
    #     print(batch)
    #     break       # 只看1批数据即可.


    # 3. 测试模型结构.
    # 会报警告, 因为你当前用的PyTorch版本没有编译'flash attention'加速功能.  'flash attention'是斯坦福大学提出的一种 高效注意力计算优化技术.
    # 如果你追求极致的性能, 可以尝试: 安装编译了'flash attention'的PyTorch版本, 但是步骤复杂: GPU, CUDA版, 从源码编译PyTorch, 且windows环境配置难度更高.
    # use_AiModel()

    # 4. 模型训练.
    # train_model()

    # 5. 模型评估.
    evaluate_model()