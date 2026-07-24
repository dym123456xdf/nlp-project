"""
04_model_gru.py
================

对应原文件 todo 6.3: 搭建 GRU 网络.

GRU 与 RNN 的关系:
    GRU 是 RNN 的简化升级版, 只有 2 个门 (重置门 r / 更新门 z),
    没有 LSTM 的细胞状态 c. 因此前向签名和 RNN 完全一致, 直接改类名即可.
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
# todo 6.3 GRU 模型类
# ============================================================
class My_GRU(nn.Module):
    """
    一个最简 GRU 分类器, 用于对人名做国家分类.

    参数同 My_RNN.
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # GRU 层
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers)

        # 全连接层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # 激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播: 与 RNN 几乎一致, 不需要 c
    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        output, hn = self.rnn(input, hidden)
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hn

    # 初始化隐藏状态: 与 RNN 一致
    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('todo 6.3 验证: GRU 模型')
    print('=' * 60)

    model = My_GRU(n_letters, 128, category_num)
    print(f'GRU 模型结构: \n{model}\n')

    print('--- 喂 1 个随机人名走一遍 ---')
    input_tensor = torch.randn(6, n_letters)
    hidden = model.init_hidden()
    output, hn = model(input_tensor, hidden)

    print(f'输入形状:    {input_tensor.shape}')
    print(f'输出形状:    {output.shape}')      # [1, 18]
    print(f'输出内容:    {output}')
    print(f'隐藏状态形状: {hn.shape}')          # [1, 1, 128]
