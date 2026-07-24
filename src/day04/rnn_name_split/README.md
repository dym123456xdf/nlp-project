# RNN 全球人名分类案例 - 拆分版

本目录把 `dm01_RNN全球人名分类案例.py` 拆分成多个独立文件,
每一部分单独成一个 `.py`, 既方便单独理解、单独运行, 也方便对比查看。

> 文件名按阅读顺序加了数字前缀 (00 / 01 / ...), 看起来一目了然.
> 因为 Python 标识符不能以数字开头, 所以 `from common import ...` 这类写法
> 需要 1 个运行时 hook 帮忙把不带编号的名字解析到带编号的文件, 这段 hook
> 在每个文件的 docstring 下面, 看得到 (可阅读, 正常运行, 不要删).

## 推荐阅读顺序

| # | 文件名 | 对应原代码 todo | 功能 |
|---|--------|----------------|------|
| 00 | `00_common.py` | 顶部导包 / 全局常量 | 公共代码: 字母表 / 国家列表 / 字体设置 |
| 01 | `01_data_preparation.py` | todo 1 / 2 / 3 / 4 / 5 | 数据加载 + 数据集 + 数据加载器 |
| 02 | `02_model_rnn.py` | todo 6.1 + 7.1 | RNN 模型类 + 单独测试 |
| 03 | `03_model_lstm.py` | todo 6.2 | LSTM 模型类 |
| 04 | `04_model_gru.py` | todo 6.3 | GRU 模型类 |
| 05 | `05_test_rnn_lstm_gru.py` | todo 7.2 | RNN/LSTM/GRU 三模型对比测试 |
| 06 | `06_train_common.py` | (共享基础设施) | 三模型训练循环: `train_one_model()` |
| 07 | `07_train_rnn.py` | todo 8.1 | RNN 训练 (~7s, 1 epoch) |
| 08 | `08_train_lstm.py` | todo 8.2 | LSTM 训练 (~16s, 1 epoch) |
| 09 | `09_train_gru.py` | todo 8.3 | GRU 训练 (~16s, 1 epoch) |
| 10 | `10_compare_plot.py` | todo 9 | 三模型损失 / 耗时 / 准确率绘图 |
| 11 | `11_predict.py` | todo 10 | RNN/LSTM/GRU 模型预测 |
| 99 | `99_main.py` | - | 一键串联: 数据 → 训练 → 绘图 → 预测 |

## 运行方式

### 方式 1: 单独看某一部分 (推荐初学, 容易懂)

```bash
cd src/day04/rnn_name_split
python 01_data_preparation.py    # 数据准备
python 02_model_rnn.py           # 看 RNN 模型
python 07_train_rnn.py           # 训练 RNN (~7s)
python 11_predict.py             # 拿已训好的 RNN 模型预测
```

### 方式 2: 一键串联 (复刻原文件 `if __name__ == '__main__'`)

```bash
cd src/day04/rnn_name_split
python 99_main.py                # 全跑: 数据+训练+预测 (默认跳过 plt.show)
RUN_PLOT=1 python 99_main.py     # 加上绘图 (会卡住等你点窗口)
```

### 跳过某些阶段

通过环境变量控制, 例如只跑预测不训练:

```bash
RUN_TRAIN=0 RUN_PREDICT=1 python 99_main.py
```

| 环境变量 | 默认 | 作用 |
|----------|------|------|
| `RUN_LOAD`    | 1 | 数据加载器自检 |
| `RUN_TRAIN`   | 1 | 训练 RNN/LSTM/GRU |
| `RUN_PLOT`    | 0 | 绘图 (要手动关 plt 窗口, 默认关) |
| `RUN_PREDICT` | 1 | 模型预测 |

## 目录约定

- 数据: `../data/name_classfication.txt` (20074 条人名 + 18 个国家)
- 模型: `../model/`
- 图片: `../img/`

## 与原文件对应关系

完全按照原文件 `dm01_RNN全球人名分类案例.py` 的代码块一一对应拆分,
拆完后逐文件验证可运行 (13 个 .py 全部能跑).

本目录不会改动 `dm01_RNN全球人名分类案例.py`, 二者并存.

## 数字前缀对 import 的影响: 一次说清

- 标识符(`from X import Y` 中的 `X`) **不能以数字开头**, 所以不能用
  `from 00_common import ...` 这种写法.
- 解决: 每个文件 docstring 下面有 1 个 `sys.meta_path` hook,
  当 Python 去找名为 `common` 的模块时, 这个 hook 会把它重定向到文件
  `00_common.py`. 这样 import 写法和原文件保持一致 (`from common import ...`),
  文件名也保留了排序号.

如果不想用 hook, 手动方案:

```bash
mv 00_common.py common.py
# 同时把每个文件里的 `from common import ...` 之类的 import 都不变(本来就不变)
```
