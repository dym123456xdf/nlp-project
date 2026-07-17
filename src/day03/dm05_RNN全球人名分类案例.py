"""
案例:
    代码实现RNN 全球人名分类案例, 录入人名, 预测其国家.

回顾: 深度学习, NLP项目研发流程:
    1. 导包
    2. 数据的预处理.
        文件加载 -> 封装成Tensor -> 数据集对象(TensorDataset) -> 数据加载器(DataLoader)
    3. 构建模型.
        RNN, LSTM, GRU
    4. 模型训练.
        绘图 -> 对比三种模型的效果.
    5. 模型测试.
        RNN, LSTM, GRU

细节:
    1. 本案例是(课程中, NLP阶段)为数不多的 用one-hot编码来处理的案例.
    2. 本案例是(课程中, NLP阶段)为数不多的 用RNN, LSTM, GRU三种模型全演示的案例.
    3. 代码层次上, 优先掌握: LSTM, RNN, 因为GRU写法和RNN几乎一致, 简单改改即可.
"""

# 导包
import torch                                        # 张量计算相关
import torch.nn as nn                               # 神经网络模块, 各种模型的层, 组件...
import torch.nn.functional as F                     # 常用的函数库...
import torch.optim as optim                         # 优化器模块
from  torch.utils.data import Dataset, DataLoader   # 数据集对象, 数据加载器
import string                                       # 字符串处理模块.
import time                                         # 时间模块.
import matplotlib.pyplot as plt                     # 绘图模块.
from tqdm import tqdm                               # 进度条

# 解决绘图时, 中文乱码问题.
plt.rcParams['font.sans-serif'] = ['SimHei']        # Mac本换成: 'Arial Unicode  MS'
plt.rcParams['axes.unicode_minus'] = False


# todo 1. 定义遍历, 获取常用的字符数量.
# 1. 获取所有的常用字符 -> 包括 字母 + 符号
all_letters = string.ascii_letters + " .,;'"        # 52个字母(大小写形式) + '空格 点 逗号 分号 单引号'
# 2. 获取常用的字符的数量
n_letters = len(all_letters)

# print('所有常用字符: ', all_letters)          # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
# print('常用字符数量: ', n_letters)            # 57


# todo 2. 定义遍历, 获取常用国家名 种类数 和 个数.
# 1. 国家名 种类数.
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 2. 国家名 个数.
category_num = len(categories)

print('国家名: ', categories)
print('国家名种类数: ', category_num)         # 18个国家名


# todo 3. 定义函数, 实现: 读取源数据到内存.
def read_data(file_path):
    """
    读取源数据到内存中, 并把 特征(人名) 和 标签(国家) 分别存储到两个列表中.
    :param file_path: 源数据文件的路径
    :return: my_list_x: 存储的人名(特征), my_list_y: 存储的国家名(标签)
    """
    # 1. 创建两个列表, 分别存储: 人名(特征), 国家名(标签)
    my_list_x, my_list_y = [], []

    # 2. 关联文件, 并读取其内容(逐行读取)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 3. 遍历, 获取到每一行的数据.
        for line in f.readlines():
            # 4. 过滤无效数据, 假设(整行的)长度 小于等于5, 就过滤掉.  整行长度 = 人名 + '\t' + 国家名
            if len(line) <= 5:
                continue
            # 5. 走到这里, 说明该行数据是有效数据(即: 行长度 > 5), 处理后, 添加到对应的列表中.
            x, y = line.strip().split('\t')
            # 6. 添加到对应的列表中.
            my_list_x.append(x)
            my_list_y.append(y)

    # 扩展: 查看下数据集的长度, 即: 判断数据是否正确.
    print(f'my_list_x: {len(my_list_x)}')       # 20074
    print(f'my_list_y: {len(my_list_y)}')       # 20074

    # 7. 返回解析后的 样本 和 标签.
    return my_list_x, my_list_y


# todo 4. 创建数据集对象, 即: 原始数据 -> 数据集对象TensorDataset -> 数据加载器DataLoader
class NameClassDataset(Dataset):
    # 1. 初始化函数, 接收: 样本和标签数据, 初始化数据集基本属性.
    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x          # 存储样本数据列表
        self.my_list_y = my_list_y          # 存储标签数据列表
        self.sample_len = len(my_list_x)    # 计算样本总数并存储, 20074

    # 2. 定义函数, 用于获取样本总数.    外界用 len(NameClassDataset对象) 的时候, 自动触发.
    def __len__(self):
        return self.sample_len


    # 3. 定义函数, 实现根据指定索引, 获取其对应的样本.
    def __getitem__(self, index):
        """
        根据指定的索引, 获取其对应的样本, 并进行 one-hot编码 和 张量转换.
        :param index: 样本索引
        :return:  tensor_x: 人名(特征)的one-hot编码,  tensor_y: 国家(标签)的张量表示
        """
        # 1. 索引边界校验, 确保索引在合法范围.    [0, self.sample_len - 1]
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按照索引获取原始样本 和 标签.
        x = self.my_list_x[index]       # 例如: Ding      ->  (4, 57)
        y = self.my_list_y[index]       # 例如: Chinese   ->  18个国家中的某个索引, 例如: 6

        # 3. 人名数据转换为 one-hot编码.
        # 3.1 生成全0张量
        tensor_x = torch.zeros(len(x), n_letters)       # 例如: [4, 57]
        # 3.2 遍历人名, 获取每个字母, 生成one-hot张量.
        for li, letter in enumerate(x):
            # 3.2.1 获取字母在 全局字母表中的索引位置, 例如: 字母'D' 在 "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'"中的位置
            letter_index = all_letters.find(letter)
            # 3.2.2 在对应位置设置为1 -> 即: one-hot编码
            tensor_x[li][letter_index] = 1

        # 4. 国家数据转换为 张量.
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)

        # 5. 返回结果
        return tensor_x, tensor_y


# todo 5. 定义函数, 获取数据加载器对象 -> 思路: Tensor -> TensorDataset -> DataLoader
def get_dataloader():
    # 1. 读取数据文件, 获取: 样本(人名)列表 和 标签(国家名)列表.
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 3. 创建数据加载器对象, 用于批量加载和处理数据.
    # 参1: 数据集对象(20074个人名 和 国家名),  参2: 批次大小, 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

    # 4. 测试数据加载器, 打印第一批数据(某一个样本的) 形状 和 内容
    for x, y in my_dataloader:
        print(f'x.shape: {x.shape}, x: {x}')        # 人名的张量形状和内容.
        print(f'y.shape: {y.shape}, y: {y}')        # 国家张量形状和内容.
        break       # 仅打印第1批次数据, 用于查看, 避免全部输出.

    # 5. 优化1: 可以把上述的数据加载器给返回, 后续直接调用.
    # return my_dataloader


# todo n. 测试代码
if __name__ == '__main__':
    # 1. 读取数据
    # my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 2. 测试: 数据加载器.
    get_dataloader()