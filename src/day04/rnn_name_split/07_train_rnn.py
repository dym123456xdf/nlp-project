"""
07_train_rnn.py
================

对应原文件 todo 8.1: 用 My_RNN 训练 1 轮.

调用方式:
    cd src/day04/rnn_name_split
    python 07_train_rnn.py
"""

# 导入


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

from model_rnn import My_RNN
from train_common import train_one_model


# 直接训练并打印 / 返回指标
if __name__ == '__main__':
    print('=' * 60)
    print('todo 8.1 RNN 模型训练')
    print('=' * 60)
    loss_list, total_time, acc_list = train_one_model(My_RNN, 'rnn')
    print(f'\n最终返回:')
    print(f'  损失列表长度: {len(loss_list)}')
    print(f'  总耗时: {total_time}s')
    print(f'  准确率列表长度: {len(acc_list)}')
