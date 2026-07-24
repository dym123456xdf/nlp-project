"""
99_main.py
==========

一键串联: 数据 -> 训练 -> 绘图 -> 预测.
对应原文件 `dm01_RNN全球人名分类案例.py` 末尾的 `if __name__ == '__main__':` 区.

默认会跑:
    1. 数据加载器自检
    2. RNN / LSTM / GRU 模型训练
    3. 三模型损失 / 耗时 / 准确率绘图 (本地 plt.show, 不会自动关闭)
    4. RNN / LSTM / GRU 预测 3 个测试名字

如果想跳过训练阶段(只跑预测), 可以这样:
    RUN_TRAIN=False RUN_PLOT=False RUN_PREDICT=True python main.py
"""



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

from common import plt

# 数据预处理
from data_preparation import get_loader

# 三个模型类
from model_rnn import My_RNN
from model_lstm import My_LSTM
from model_gru import My_GRU

# 训练 / 绘图 / 预测函数
from train_common import train_one_model
from compare_plot import compare_plot
from predict import dm_predict_rnn, dm_predict_lstm, dm_predict_gru


# ============================================================
# 总开关: 用环境变量控制阶段, 方便单独跑某一步
# ============================================================
RUN_LOAD    = os.environ.get('RUN_LOAD',    '1') == '1'        # 默认开
RUN_TRAIN   = os.environ.get('RUN_TRAIN',   '1') == '1'        # 默认开
RUN_PLOT    = os.environ.get('RUN_PLOT',    '0') == '1'        # 默认关(避免画图等待键盘)
RUN_PREDICT = os.environ.get('RUN_PREDICT', '1') == '1'        # 默认开


def main():
    print('=' * 60)
    print('RNN 全球人名分类案例 - 拆分版  入口')
    print('=' * 60)

    # ============================================================
    # 1. 数据加载器自检
    # ============================================================
    if RUN_LOAD:
        print('\n[1] 数据加载器自检(取 1 个 batch 看看)')
        loader = get_loader(batch_size=1, shuffle=True)
        for x, y in loader:
            print(f'  x.shape: {x.shape}, y: {y.item()}')
            break

    # ============================================================
    # 2. 训练 RNN / LSTM / GRU
    # ============================================================
    if RUN_TRAIN:
        print('\n[2] 模型训练阶段')
        rnn_metrics  = train_one_model(My_RNN, 'rnn')
        lstm_metrics = train_one_model(My_LSTM, 'lstm')
        gru_metrics  = train_one_model(My_GRU, 'gru')
        print('\n训练阶段完成.')
    else:
        rnn_metrics = lstm_metrics = gru_metrics = None

    # ============================================================
    # 3. 绘制三模型对比图
    # ============================================================
    if RUN_PLOT:
        print('\n[3] 三模型绘图(会调用 plt.show, 需要手动关闭窗口)')
        compare_plot()
    else:
        print('\n[3] 跳过绘图阶段(若需开启: RUN_PLOT=1 python 99_main.py)')

    # ============================================================
    # 4. 模型预测
    # ============================================================
    if RUN_PREDICT:
        print('\n[4] 模型预测')
        # 试 3 个有代表性的名字
        test_names = ['Piao', 'Zhang', 'Smith']
        for name in test_names:
            dm_predict_rnn(name)
            dm_predict_lstm(name)
            dm_predict_gru(name)

    print('\n' + '=' * 60)
    print('拆分版 main 跑完.')
    print('=' * 60)


if __name__ == '__main__':
    main()
