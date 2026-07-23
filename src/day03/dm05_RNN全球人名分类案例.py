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
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']    # Mac本用: 'Arial Unicode  MS'
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
def my_collate_fn(batch):
    """
    自定义 collate_fn: 处理变长人名序列.
    默认 collate 会对变长 tensor 做 stack, 长度不齐会报错.
    这里改用 nn.utils.rnn.pad_sequence 把一个 batch 内所有人名补成同样长度.
    返回 padded tensor 而不是 PackedSequence, 是为了让 forward 更直观 (课程级).

    :param batch: list of (tensor_x, tensor_y), 每个 tensor_x 形状 (seq_len_i, 57)
    :return: x_padded: (max_seq_len, batch, 57)
             y:        (batch,)
             seq_lengths: (batch,)  各样本真实长度, 后续若想升级 PackedSequence 可用
    """
    # 1. 拆分 batch 里的样本和标签
    xs, ys = list(zip(*batch))
    # 2. 用 pad_sequence 把不同长度的人名补成等长.
    #    batch_first=False: 拼成 (max_seq_len, batch, 57); 补的位是 0 张量 (one-hot 全 0 不影响类别).
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=False)
    # 3. 记录每个样本的真实长度 (调试用, 不影响当前 forward)
    seq_lengths = torch.tensor([len(x) for x in xs])
    # 4. 标签 stack 成 (batch,)
    y = torch.stack(ys, dim=0)
    return x_padded, y


def get_dataloader(batch_size=64):
    """
    读取数据, 构建数据集对象, 并返回数据加载器.
    :param batch_size: 批次大小
    :return: my_dataloader
    """
    # 1. 读取数据文件, 获取: 样本(人名)列表 和 标签(国家名)列表.
    #    用绝对路径, 无论从哪启动都能找到数据.
    import os
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'name_classfication.txt')
    my_list_x, my_list_y = read_data(data_path)

    # 2. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 3. 创建数据加载器对象, 用于批量加载和处理数据.
    #    collate_fn 必须用自定义的, 因为每个人名长度不一样, 默认 collate 会报错.
    my_dataloader = DataLoader(name_class_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

    # 4. 返回数据加载器.
    return my_dataloader


# ========================== 补充: 模型 / 训练 / 绘图 / 测试 ==========================

# todo 6. 构建RNN家族模型 (RNN, LSTM, GRU 三选一, 通过 model_type 切换)
class NameClassifyModel(nn.Module):
    """
    人名 -> 国家 分类模型.
    模型结构:  one-hot输入 -> (RNN / LSTM / GRU) -> 取最后时间步隐藏状态 -> Linear -> 18类.
    """
    def __init__(self, model_type='rnn', input_size=n_letters, hidden_size=128, num_layers=2, output_size=category_num):
        super().__init__()
        # 1. 保存核心超参
        self.model_type = model_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 2. 根据 model_type 选择不同的循环层
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f'model_type 仅支持 rnn / lstm / gru, 当前传入: {model_type}')

        # 3. 输出层: hidden_size -> 类别数
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        :param x: (seq_len, batch_size, input_size)  即: (人名字符数, 批次, 57)
        :return:  (batch_size, output_size)
        """
        # 1. 初始化隐藏状态; LSTM 还要再给一个 cell state
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if self.model_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
            # output: (seq_len, batch, hidden_size);  (hn, cn): (num_layers, batch, hidden_size)
            output, (hn, cn) = self.rnn(x, (h0, c0))
        else:
            # output, hn: (seq_len, batch, hidden_size);  hn: (num_layers, batch, hidden_size)
            output, hn = self.rnn(x, h0)

        # 2. 取最后一层最后一个时间步的隐藏状态 作为人名整体语义表示
        #    hn[-1] 形状: (batch, hidden_size)
        final_hidden = hn[-1]

        # 3. 送入全连接层, 得到各类别 logits
        out = self.fc(final_hidden)        # (batch, output_size=18)
        return out


# todo 7. 模型训练函数 -> 三种模型各跑一遍, 收集每轮的 loss.
def train_model(model, dataloader, epochs=5, lr=0.005):
    """
    :param model: 待训练模型
    :param dataloader: 数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :return: loss_list: 每个 epoch 的平均 loss
    """
    # 1. 设备选择: mac 上有 mps 就用 mps, 否则 cpu
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    # 2. 损失函数 + 优化器
    criterion = nn.CrossEntropyLoss()                  # 交叉熵损失 (自带 softmax, 输入 logits 即可)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 收敛比 SGD 稳定

    # 3. 记录每个 epoch 的平均 loss, 用于后续绘图
    loss_list = []

    # 4. 开始训练
    for epoch in range(1, epochs + 1):
        model.train()                 # 训练模式 (对 Dropout/BN 生效, 本模型暂无, 但保持规范)
        total_loss = 0.0              # 当前 epoch 累计 loss
        start_time = time.time()      # 计时, 看每轮耗时
        # tqdm 展示进度条
        bar = tqdm(dataloader, desc=f'Epoch {epoch}/{epochs}', ncols=80)
        for x, y in bar:
            # x: (seq_len, batch, 57)   y: (batch,)
            x, y = x.to(device), y.to(device)
            # 梯度清零 -> 前向 -> 计算损失 -> 反向 -> 更新参数
            optimizer.zero_grad()
            outputs = model(x)                        # (batch, 18)
            loss = criterion(outputs, y)
            loss.backward()
            # 梯度裁剪, 防止 RNN 家族在长序列上梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f'>>> Epoch {epoch}/{epochs}  avg_loss: {avg_loss:.4f}  耗时: {time.time() - start_time:.1f}s')

    return loss_list


# todo 8. 绘图函数 -> 把三种模型的 loss 曲线画到一张图上对比.
def plot_loss(rnn_loss, lstm_loss, gru_loss):
    """
    :param rnn_loss:   RNN  训练 loss 列表
    :param lstm_loss:  LSTM 训练 loss 列表
    :param gru_loss:   GRU  训练 loss 列表
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(rnn_loss) + 1),  rnn_loss,  marker='o', label='RNN')
    plt.plot(range(1, len(lstm_loss) + 1), lstm_loss, marker='s', label='LSTM')
    plt.plot(range(1, len(gru_loss) + 1),  gru_loss,  marker='^', label='GRU')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN vs LSTM vs GRU  人名分类 loss 对比')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    # 保存 + 展示 (在 PyCharm 里 plt.show() 会弹窗)
    plt.savefig('./rnn_lstm_gru_loss.png', dpi=120)
    print('loss 曲线已保存到: ./rnn_lstm_gru_loss.png')
    plt.show()


# todo 9. 模型测试函数 -> 给定人名列表, 打印预测国家 + 真实国家.
def predict(model, names, real_categories=None):
    """
    :param model: 训练好的模型
    :param names: 待预测的人名列表, 例如 ['Ding', 'Smith', 'Ivanov']
    :param real_categories: 对应的真实国家 (可选), 传入后可一并打印
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for i, name in enumerate(names):
            # 1. 把人名转成 one-hot: (seq_len, 1, 57)
            tensor_x = torch.zeros(len(name), 1, n_letters)
            for li, letter in enumerate(name):
                li_idx = all_letters.find(letter)
                if li_idx == -1:
                    # 字符不在字母表里 (例如中文人名), 跳过该字符
                    continue
                tensor_x[li][0][li_idx] = 1
            tensor_x = tensor_x.to(device)

            # 2. 模型预测, 取 logits 最大索引 -> 国家
            output = model(tensor_x)                       # (1, 18)
            pred_idx = output.argmax(dim=1).item()
            pred_country = categories[pred_idx]

            # 3. 打印
            real_info = ''
            if real_categories is not None and i < len(real_categories):
                real_info = f'  真实: {real_categories[i]}'
            print(f'人名: {name:<10}  预测国家: {pred_country:<12}{real_info}')


# todo n. 测试代码
if __name__ == '__main__':
    # ----------------- 阶段1: 数据加载 -----------------
    print('=' * 60)
    print('阶段1: 数据加载')
    print('=' * 60)
    train_loader = get_dataloader(batch_size=64)

    # ----------------- 阶段2: 训练三种模型 -----------------
    EPOCHS = 5                # 训练轮数, 可按机器性能调整
    LR = 0.005                # 学习率

    # RNN
    print('\n' + '=' * 60)
    print('阶段2-1: 训练 RNN')
    print('=' * 60)
    rnn_model  = NameClassifyModel(model_type='rnn')
    print(rnn_model)
    rnn_loss   = train_model(rnn_model, train_loader, epochs=EPOCHS, lr=LR)

    # LSTM
    print('\n' + '=' * 60)
    print('阶段2-2: 训练 LSTM')
    print('=' * 60)
    lstm_model = NameClassifyModel(model_type='lstm')
    print(lstm_model)
    lstm_loss  = train_model(lstm_model, train_loader, epochs=EPOCHS, lr=LR)

    # GRU
    print('\n' + '=' * 60)
    print('阶段2-3: 训练 GRU')
    print('=' * 60)
    gru_model  = NameClassifyModel(model_type='gru')
    print(gru_model)
    gru_loss   = train_model(gru_model, train_loader, epochs=EPOCHS, lr=LR)

    # ----------------- 阶段3: 绘制 loss 对比图 -----------------
    print('\n' + '=' * 60)
    print('阶段3: 绘制 loss 对比图')
    print('=' * 60)
    plot_loss(rnn_loss, lstm_loss, gru_loss)

    # ----------------- 阶段4: 用训练好的模型做预测 -----------------
    print('\n' + '=' * 60)
    print('阶段4: 模型测试 (以 LSTM 为例)')
    print('=' * 60)
    test_names = ['Ding', 'Smith', 'Ivanov', 'Garcia', 'Mueller', 'Tanaka', 'Nguyen', 'O\'Brien']
    predict(lstm_model, test_names)

    print('\n' + '=' * 60)
    print('全部完成')
    print('=' * 60)