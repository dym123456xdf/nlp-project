"""
05_test_rnn_lstm_gru.py
========================

对应原文件 todo 7.2: 用真实的 1 条数据, 跑一遍 RNN / LSTM / GRU, 看三种模型输出维度.

这个测试的目的:
    1. 验证数据加载器拿出来的样本 形状 跟模型期望的输入形状能对上.
    2. 对照三种模型的 forward 写法差异(尤其是 LSTM 的 c).
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
    torch, DataLoader,
    n_letters, category_num,
)
from data_preparation import NameClassDataset, read_data, DATA_FILE
from model_rnn import My_RNN
from model_lstm import My_LSTM
from model_gru import My_GRU


# ============================================================
# todo 7.2 三模型对比测试
# ============================================================
def dm_test_rnn_lstm_gru():
    """
    用真实数据(每种模型各跑 1 个样本), 对比三种模型的:
        输入张量形状, 输出张量形状, 输出的具体数值.
    """

    # 1. 模型超参 (方便跟训练阶段共享)
    input_size, n_hidden, output_size = n_letters, 128, category_num       # 57, 128, 18

    # 2. 读数据 + 建数据集 + 建 DataLoader
    my_list_x, my_list_y = read_data(DATA_FILE)
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)
    my_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

    # 3. 模型初始化
    my_rnn = My_RNN(input_size, n_hidden, output_size)
    my_lstm = My_LSTM(input_size, n_hidden, output_size)
    my_gru = My_GRU(input_size, n_hidden, output_size)

    # 4. 打印模型结构
    print('--- 模型结构 ---')
    print(f'RNN : {my_rnn}')
    print(f'LSTM: {my_lstm}')
    print(f'GRU : {my_gru}\n')

    # ====================================================
    # 7.1 测试 RNN
    # ====================================================
    print('=' * 60)
    print('测试 1: RNN')
    print('=' * 60)
    for i, (x, y) in enumerate(my_dataloader):
        print(f'i(样本编号): {i}')
        print(f'x(输入人名 one-hot 形状): {x.shape}')     # [1, seq_len, 57]
        print(f'y(国家索引): {y}, 形状: {y.shape}')     # 形如 tensor([15])

        hidden = my_rnn.init_hidden()                  # [1, 1, 128]
        output, hidden = my_rnn(x[0], hidden)          # x[0] -> [seq_len, 57]
        print(f'RNN 输出形状: {output.shape}')          # [1, 18]
        print(f'RNN 预测结果(对数概率): \n{output}')
        break

    # ====================================================
    # 7.2 测试 LSTM
    # ====================================================
    print('\n' + '=' * 60)
    print('测试 2: LSTM')
    print('=' * 60)
    for i, (x, y) in enumerate(my_dataloader):
        hidden, c = my_lstm.init_hidden()              # 各返回 [1, 1, 128]
        output, hidden, c = my_lstm(x[0], hidden, c)
        print(f'i: {i}')
        print(f'x: {x.shape},  y: {y.shape}')
        print(f'LSTM 输出形状: {output.shape}')          # [1, 18]
        print(f'LSTM 预测结果: \n{output}')
        break

    # ====================================================
    # 7.3 测试 GRU
    # ====================================================
    print('\n' + '=' * 60)
    print('测试 3: GRU')
    print('=' * 60)
    for i, (x, y) in enumerate(my_dataloader):
        hidden = my_gru.init_hidden()
        output, hidden = my_gru(x[0], hidden)
        print(f'i: {i}')
        print(f'x: {x.shape},  y: {y.shape}')
        print(f'GRU 输出形状: {output.shape}')
        print(f'GRU 预测结果: \n{output}')
        break


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    dm_test_rnn_lstm_gru()
