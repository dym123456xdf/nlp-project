"""
06_train_common.py
====================

为 train_rnn / train_lstm / train_gru 三个训练脚本提供共享的训练循环.

把"数据加载 + 模型构建 + 训练循环 + 统计指标"封装到 1 个函数,
三个 07/08/09_xxx.py 只负责传入对应的模型类即可.
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
    torch, nn, optim, DataLoader, time, tqdm,
    n_letters, category_num,
)
from data_preparation import NameClassDataset, read_data, DATA_FILE


# ============================================================
# 超参
# ============================================================
LEARNING_RATE = 1e-3
EPOCHS = 1


def train_one_model(model_cls, model_name: str, hidden_size: int = 128):
    """
    通用训练循环, 用 batch_size=1 训练 1 个 epoch.

    :param model_cls:    模型类, 取值 My_RNN / My_LSTM / My_GRU
    :param model_name:   模型名, 用于打印 & 拼接模型文件名
    :param hidden_size:  隐藏层维度
    :return: (loss_list, total_time, acc_list)
             loss_list: 每 100 个样本求一次平均损失
             acc_list:  每 100 个样本求一次平均准确率
             total_time: 总训练耗时(秒)
    """
    # 1. 数据准备
    my_list_x, my_list_y = read_data(DATA_FILE)
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 2. 构造模型
    input_size, output_size = n_letters, category_num
    my_model = model_cls(input_size, hidden_size, output_size)

    # 3. 损失函数 + 优化器
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(my_model.parameters(), lr=LEARNING_RATE)

    # 4. 训练过程 -> 参数初始化
    start_time = time.time()
    total_iter_num = 0              # 已训练样本数
    total_loss = 0.0                # 累计损失
    loss_list = []                  # 每 100 步的平均损失
    total_acc_num = 0               # 已预测正确的样本数
    acc_list = []                   # 每 100 步的平均准确率

    # 5. 按 epoch 遍历
    for epoch in range(EPOCHS):
        print(f'\n开始第 {epoch + 1}/{EPOCHS} 轮训练({model_name})...')
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            # 5.1 前向传播
            #      针对 RNN / GRU, init_hidden 只返回 hidden
            #      针对 LSTM, init_hidden 返回 (hidden, c)
            #      所以这里用一个 try / except / 自适应判定
            init = my_model.init_hidden()
            if isinstance(init, tuple):                                   # LSTM
                hidden, c = init
                output, hidden, c = my_model(x[0], hidden, c)
            else:                                                          # RNN / GRU
                output, hidden = my_model(x[0], init)

            # 5.2 算损失
            my_loss = criterion(output, y)

            # 5.3 三剑客
            optimizer.zero_grad()
            my_loss.backward()
            optimizer.step()

            # 5.4 统计
            total_iter_num += 1
            total_loss += my_loss.item()

            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y.item() else 0)

            # 5.5 每 100 步记一次平均
            if total_iter_num % 100 == 0:
                avg_loss = total_loss / total_iter_num
                avg_acc = total_acc_num / total_iter_num
                loss_list.append(avg_loss)
                acc_list.append(avg_acc)

            # 5.6 每 2000 步打印一次日志
            if total_iter_num % 2000 == 0:
                avg_loss = total_loss / total_iter_num
                avg_acc = total_acc_num / total_iter_num
                elapsed = int(time.time() - start_time)
                print(f'\n轮次: {epoch + 1}, 训练样本数: {total_iter_num}, '
                      f'平均损失: {avg_loss:.4f}, 耗时: {elapsed}s, 准确率: {avg_acc:.4f}')

        # 6. 一轮结束, 保存模型参数
        save_path = f'../model/my_{model_name}_wh02_{epoch + 1}.bin'
        torch.save(my_model.state_dict(), save_path)
        print(f'已保存模型参数: {save_path}')

    # 7. 训练结束
    total_time = int(time.time() - start_time)
    print(f'\n[{model_name}] 训练完成, 总耗时: {total_time}s, '
          f'总训练了 {total_iter_num} 个样本!!')

    return loss_list, total_time, acc_list
