"""
11_predict.py
=============

对应原文件 todo 10.1 / 10.2 / 10.3: 用训练好的模型预测一个新的人名.

前置条件:
    已经跑过 07_train_rnn.py / 08_train_lstm.py / 09_train_gru.py,
    在 ../model/ 下生成了:
        my_rnn_wh02_1.bin
        my_lstm_wh02_1.bin
        my_gru_wh02_1.bin

用法:
    cd src/day04/rnn_name_split
    python 11_predict.py
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

import os
from common import (
    torch, n_letters, category_num, categories, all_letters,
)
from model_rnn import My_RNN
from model_lstm import My_LSTM
from model_gru import My_GRU


# 三个模型参数文件路径
MODEL_PATHS = {
    'rnn':  '../model/my_rnn_wh02_1.bin',
    'lstm': '../model/my_lstm_wh02_1.bin',
    'gru':  '../model/my_gru_wh02_1.bin',
}


# ============================================================
# todo 10.1.2 把人名 转成 one-hot 张量, 形状 [seq_len, n_letters]
# ============================================================
def line_to_tensor(line: str):
    """把人名(字符串) 转成 one-hot 编码, 形状 [len, n_letters]."""
    tensor_x = torch.zeros(len(line), n_letters)
    for i, letter in enumerate(line):
        letter_index = all_letters.find(letter)
        tensor_x[i][letter_index] = 1
    return tensor_x


# ============================================================
# 通用预测函数: 适配三种模型
# ============================================================
def predict_one(model_cls, model_key: str, name: str, top_k: int = 3):
    """
    用训练好的某个模型预测人名对应的国家, 返回 top_k 候选.

    :param model_cls: 模型类 (My_RNN / My_LSTM / My_GRU)
    :param model_key: 'rnn' / 'lstm' / 'gru', 用于找参数文件
    :param name:      待预测的人名字符串
    :param top_k:     返回 top-k 个候选, 默认 3
    :return: [(category_name, log_prob_value), ...] 长度 = top_k
    """
    # 1. 实例化模型 + 加载权重
    model = model_cls(n_letters, 128, category_num)
    model.load_state_dict(torch.load(MODEL_PATHS[model_key], map_location='cpu'))
    model.eval()

    # 2. one-hot + 推理
    x_tensor = line_to_tensor(name)
    with torch.no_grad():
        init = model.init_hidden()
        # LSTM: init 是 (hidden, c) tuple, forward 返回 (out, h, c)
        # RNN/GRU: init 是 tensor, forward 返回 (out, h)
        if isinstance(init, tuple):
            hidden, c = init
            output, _, _ = model(x_tensor, hidden, c)
        else:
            output, _ = model(x_tensor, init)

        # 3. 取 top-k
        topv, topi = output.topk(top_k, 1, True)
        return [
            (categories[topi[0][i].item()], topv[0][i].item())
            for i in range(top_k)
        ]


# ============================================================
# todo 10.1.3 单模型预测 + 打印 top-3
# ============================================================
def dm_predict_rnn(x: str):
    """只用 RNN 模型做预测."""
    print(f'\n[dm_predict_rnn] 待预测人名: {x}')
    top3 = predict_one(My_RNN, 'rnn', x, top_k=3)
    for i, (cat, val) in enumerate(top3):
        print(f'  Top{i + 1}: category={cat}, log_prob={val:.4f}')


def dm_predict_lstm(x: str):
    """只用 LSTM 模型做预测."""
    print(f'\n[dm_predict_lstm] 待预测人名: {x}')
    top3 = predict_one(My_LSTM, 'lstm', x, top_k=3)
    for i, (cat, val) in enumerate(top3):
        print(f'  Top{i + 1}: category={cat}, log_prob={val:.4f}')


def dm_predict_gru(x: str):
    """只用 GRU 模型做预测."""
    print(f'\n[dm_predict_gru] 待预测人名: {x}')
    top3 = predict_one(My_GRU, 'gru', x, top_k=3)
    for i, (cat, val) in enumerate(top3):
        print(f'  Top{i + 1}: category={cat}, log_prob={val:.4f}')


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    TEST_NAMES = ['Piao', 'Zhang', 'Smith']

    print('=' * 60)
    print('todo 10: 模型预测(输入人名, 输出最可能的 Top-3 国家)')
    print('=' * 60)

    # 检查模型文件
    for key, path in MODEL_PATHS.items():
        flag = 'OK' if os.path.exists(path) else '缺失'
        print(f'[{key.upper():>4s}] {flag}: {path}')

    # 逐模型预测
    for name in TEST_NAMES:
        for key, fn in [('rnn', dm_predict_rnn), ('lstm', dm_predict_lstm), ('gru', dm_predict_gru)]:
            try:
                fn(name)
            except FileNotFoundError as e:
                print(f'  [{key}] 跳过: 模型文件不存在 ({e})')
            except Exception as e:
                print(f'  [{key}] 跳过: {e}')
