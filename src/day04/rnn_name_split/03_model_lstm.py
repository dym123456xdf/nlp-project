"""
03_model_lstm.py
=================

对应原文件 todo 6.2: 搭建 LSTM 网络.

与 RNN 的差别:
    1. nn.RNN -> nn.LSTM
    2. forward 时多传一个 c (细胞状态), 返回也多一个 c
    3. init_hidden 改成同时初始化 hidden 和 c, 各返回一个零张量.

LSTM 内部有 4 个门控 (i / f / g / o), 比 RNN 复杂, 适合长序列.
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
# todo 6.2 LSTM 模型类
# ============================================================
class My_LSTM(nn.Module):
    """
    一个最简 LSTM 分类器, 用于对人名做国家分类.

    参数同 My_RNN.
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # LSTM 层: 比 RNN 多返回一个细胞状态 c
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)

        # 全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播: 注意比 RNN 多一个 c
    def forward(self, input, hidden, c):
        """
        :param input:  [seq_len, input_size]
        :param hidden: [n_layers, batch, hidden_size]
        :param c:      [n_layers, batch, hidden_size] 细胞状态
        :return: output (对数概率), hn, cn
        """
        # 1. 加 batch 维
        input = input.unsqueeze(1)             # [seq_len, 1, input_size]

        # 2. LSTM 计算, 同时返回 hn 和 cn
        output, (hn, cn) = self.rnn(input, (hidden, c))

        # 3. 取最后 1 个时间步
        tmp_output = output[-1]

        # 4. 全连接 + LogSoftmax
        tmp_output = self.linear(tmp_output)

        return self.softmax(tmp_output), hn, cn

    # 初始化隐藏状态: hidden 和 c 都要初始化
    def init_hidden(self):
        hidden = c = torch.zeros(self.n_layers, 1, self.hidden_size)
        return hidden, c


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('todo 6.2 验证: LSTM 模型')
    print('=' * 60)

    model = My_LSTM(n_letters, 128, category_num)
    print(f'LSTM 模型结构: \n{model}\n')

    # 用一个随机张量跑一遍前向
    print('--- 喂 1 个随机人名走一遍 ---')
    input_tensor = torch.randn(6, n_letters)
    hidden, c = model.init_hidden()
    output, hn, cn = model(input_tensor, hidden, c)

    print(f'输入形状:    {input_tensor.shape}')
    print(f'输出形状:    {output.shape}')      # [1, 18]
    print(f'输出内容:    {output}')
    print(f'隐藏状态形状: {hn.shape}')          # [1, 1, 128]
    print(f'细胞状态形状: {cn.shape}')          # [1, 1, 128]
