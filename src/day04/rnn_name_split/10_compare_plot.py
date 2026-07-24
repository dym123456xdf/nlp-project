"""
10_compare_plot.py
==================

对应原文件 todo 9: 训练 3 种模型并绘图对比.

输出 3 张图, 保存到 ../img/ 下:
    1. RNN_LSTM_GRU_loss_time_split.png     三模型损失曲线
    2. RNN_LSTM_GRU_time_split.png          三模型训练耗时柱状图
    3. RNN_LSTM_GRU_acc_split.png           三模型准确率曲线

注意: 这个函数本身会跑 3 轮训练, 默认约 2~3 分钟(取决于机器).
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

from common import plt                  # matplotlib + 中文字体已在 common 设好
from model_rnn import My_RNN
from model_lstm import My_LSTM
from model_gru import My_GRU
from train_common import train_one_model


# ============================================================
# todo 9. 三模型训练 + 绘图
# ============================================================
def compare_plot():
    # 1. 训练 3 种模型, 取各自的 (loss_list, total_time, acc_list)
    print('开始训练 RNN...')
    loss_rnn, t_rnn, acc_rnn = train_one_model(My_RNN, 'rnn')

    print('\n开始训练 LSTM...')
    loss_lstm, t_lstm, acc_lstm = train_one_model(My_LSTM, 'lstm')

    print('\n开始训练 GRU...')
    loss_gru, t_gru, acc_gru = train_one_model(My_GRU, 'gru')

    # ============================================================
    # 图 1: 损失对比曲线
    # ============================================================
    plt.figure(0, figsize=(10, 5))
    plt.plot(loss_rnn, label='RNN')
    plt.plot(loss_lstm, label='LSTM')
    plt.plot(loss_gru, label='GRU')
    plt.title('模型损失对比曲线')
    plt.xlabel('训练步数(每100步)')
    plt.ylabel('平均损失值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.savefig('../img/RNN_LSTM_GRU_loss_time_split.png')
    print('已保存: ../img/RNN_LSTM_GRU_loss_time_split.png')
    plt.show()

    # ============================================================
    # 图 2: 训练耗时柱状图
    # ============================================================
    plt.figure(1, figsize=(10, 5))
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [t_rnn, t_lstm, t_gru]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.title('模型耗时对比柱状图')
    plt.savefig('../img/RNN_LSTM_GRU_time_split.png')
    print('已保存: ../img/RNN_LSTM_GRU_time_split.png')
    plt.show()

    # ============================================================
    # 图 3: 准确率对比曲线
    # ============================================================
    plt.figure(2, figsize=(10, 5))
    plt.plot(acc_rnn, label='RNN', color='red')
    plt.plot(acc_lstm, label='LSTM', color='green')
    plt.plot(acc_gru, label='GRU', color='orange')
    plt.title('模型准确率对比曲线')
    plt.legend(loc='upper left')
    plt.savefig('../img/RNN_LSTM_GRU_acc_split.png')
    print('已保存: ../img/RNN_LSTM_GRU_acc_split.png')
    plt.show()

    print('\n全部图表已生成.')


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    compare_plot()
