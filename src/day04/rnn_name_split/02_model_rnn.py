"""
02_model_rnn.py
=================

对应原文件 todo 6.1 + todo 7.1:
    todo 6.1: 搭建 RNN 网络
    todo 7.1: 用一个随机张量, 跑一遍 RNN 模型, 看输出形状.

RNN 网络结构(单层单步):
    输入 one-hot [seq_len, batch, input_size]
        |
        +-> nn.RNN(input_size, hidden_size)
        |
    输出 output(所有时间步隐藏状态), hn(最后1个时间步隐藏状态)
        |
        +-> 取最后一个时间步 output[-1]
        |
    +-> nn.Linear(hidden_size, output_size)
        |
    +-> nn.LogSoftmax(dim=-1)
        |
    输出 [batch, output_size] 的对数概率

为何用 LogSoftmax?
    原文件说明: NLLLoss 损失函数 配合 LogSoftmax = 等同于 CrossEntropyLoss.
    如果改用 CrossEntropyLoss, 这一层可以省掉.
"""

# ============================================================
# 导入
# ============================================================


# ------------------------------------------------------------------
# 兼容性 hook: Python 标识符不能以数字开头, 所以文件名是 00_common.py 但
# import 仍写 `from common import ...`. 这一段(运行时)是 sys.meta_path 钩子,
# 让不带编号的模块名能解析到带编号的文件. 不要删.
# ------------------------------------------------------------------
import os, sys

class _NumericNameFinder:
    """让 `from common import ...` 等语句找到 00_common.py (因为 Python 不允许以数字开头的标识符)."""
    _MAP = {        'common': '00_common',
                'data_preparation': '01_data_preparation',
                'model_rnn': '02_model_rnn',
                'model_lstm': '03_model_lstm',
                'model_gru': '04_model_gru',
                'train_common': '06_train_common',
                'compare_plot': '10_compare_plot',
                'predict': '11_predict'
    }
    def find_spec(self, fullname, path, target=None):
        if fullname in self._MAP and fullname not in sys.modules:
            import importlib.util
            here = os.path.dirname(os.path.abspath(__file__))
            real = self._MAP[fullname]
            spec = importlib.util.spec_from_file_location(
                fullname, os.path.join(here, real + '.py'))
            return spec
        return None

# 安装一次性 hook
if not any(isinstance(f, _NumericNameFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _NumericNameFinder())

from common import (
    torch, nn,
    n_letters, category_num
)


# ============================================================
# todo 6.1 RNN 模型类
# ============================================================
class My_RNN(nn.Module):
    """
    一个最简 RNN 分类器, 用于对人名做国家分类.

    参数:
        input_size:  输入维度 = 字母表大小 (57)
        hidden_size: 隐藏层维度, 决定模型表示能力
        output_size: 输出维度 = 国家数量 (18)
        n_layers:    RNN 层数, 默认 1
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # RNN 层: 接收输入特征, 输出隐藏状态
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers)

        # 全连接层: 把 RNN 的隐藏状态映射到国家维度
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 激活函数: LogSoftmax(NLLLoss 配对)
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播
    def forward(self, input, hidden):
        """
        :param input: [seq_len, input_size] -> 加 batch 维后 [seq_len, batch, input_size]
        :param hidden: [n_layers, batch, hidden_size]
        :return: output (对数概率), hn (最后1个时间步隐藏状态)
        """
        # 1. 加 batch 维
        input = input.unsqueeze(1)             # [seq_len, 1, input_size]

        # 2. RNN 计算
        #    output: 所有时间步的隐藏状态 形状 [seq_len, batch, hidden_size]
        #    hn:     最后 1 个时间步的隐藏状态 形状 [n_layers, batch, hidden_size]
        output, hn = self.rnn(input, hidden)

        # 3. 取出最后 1 个时间步的隐藏状态, 形状 [batch, hidden_size]
        tmp_output = output[-1]

        # 4. 全连接层升维到 output_size
        tmp_output = self.linear(tmp_output)

        # 5. LogSoftmax 取对数概率
        return self.softmax(tmp_output), hn

    # 初始化隐藏状态
    def init_hidden(self):
        """返回全 0 的初始隐藏状态, 形状 [n_layers, batch=1, hidden_size]."""
        return torch.zeros(self.n_layers, 1, self.hidden_size)


# ============================================================
# todo 7.1 用随机张量, 跑一遍模型, 看输出形状
# ============================================================
def dm_test_myrnn():
    """
    用一个随机张量模拟一个人名, 走一遍 RNN 前向传播.
    这个函数保留了原文件 todo 7.1 的全部代码, 仅做注释完善.
    """
    # 1. 实例化 RNN 对象: 输入 57, 隐藏 128, 输出 18
    my_rnn = My_RNN(n_letters, 128, category_num)
    # print(f'my_rnn: {my_rnn}')

    # 2. 构造一个随机张量当输入: 形状 [seq_len, input_size]
    #    这里 seq_len=6, 模拟 6 个字符的人名(对应"欧阳")
    input_tensor = torch.randn(6, n_letters)
    print(f'输入张量形状: {input_tensor.shape}')         # torch.Size([6, 57])

    # 3. 初始化隐藏状态(也可以写成 torch.zeros(1, 1, 128))
    h0 = my_rnn.init_hidden()

    # 4. 前向传播
    output, hn = my_rnn(input_tensor, h0)

    # 5. 打印结果
    print(f'输出形状: {output.shape}')                  # torch.Size([1, 18])
    print(f'输出内容(对数概率): {output}')
    print(f'隐藏状态形状: {hn.shape}')                  # torch.Size([1, 1, 128])


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('todo 6.1 + 7.1 验证: RNN 模型')
    print('=' * 60)

    # 打印模型结构
    model = My_RNN(n_letters, 128, category_num)
    print(f'RNN 模型结构: \n{model}\n')

    print('--- 喂 1 个随机人名走一遍 ---')
    dm_test_myrnn()
