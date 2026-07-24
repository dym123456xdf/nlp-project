"""
00_common.py
=============

公共代码模块: 提供整个拆分版所有文件都会用到的:
    1. 第三方包导入
    2. 中文字体设置 (让 matplotlib 正常显示中文)
    3. 全局常量: 字母表 / 国家列表

说明:
    原文件 dm01_RNN全球人名分类案例.py 顶部 1-75 行就是这些公共配置,
    这里统一抽出, 其它文件通过 `from common import xxx` 引用.

推荐使用方法: 其他拆分文件统一写
    from common import (
        torch, nn, F, optim,
        Dataset, DataLoader,
        string, time, plt, tqdm,
        all_letters, n_letters,
        categories, category_num,
        DATA_DIR, MODEL_DIR, IMG_DIR,
    )
"""

# ============================================================
# 1. 第三方包导入
# ============================================================
import torch                                        # 张量计算相关
import torch.nn as nn                               # 神经网络模块, 各种模型的层, 组件...
import torch.nn.functional as F                     # 常用的函数库...
import torch.optim as optim                         # 优化器模块
from torch.utils.data import Dataset, DataLoader    # 数据集对象, 数据加载器
import string                                       # 字符串处理模块
import time                                         # 时间模块
import matplotlib.pyplot as plt                     # 绘图模块
from tqdm import tqdm                               # 进度条


# ============================================================
# 2. 中文字体配置 (解决 matplotlib 中文乱码)
# ============================================================
# 原文件注释: plt.rcParams['font.sans-serif'] = ['SimHei']        # Mac本换成: 'Arial Unicode  MS'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 3. 全局常量: 字符表 / 国家列表
# ============================================================

# todo 1. 常用字符: 52个字母(大小写) + ' . , ; 单引号'
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
# all_letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;\''
# n_letters = 57


# todo 2. 国家名列表
categories = [
    'Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish',
    'Chinese', 'Vietnamese', 'Japanese', 'French', 'Greek', 'Dutch',
    'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German'
]
category_num = len(categories)        # 18


# ============================================================
# 4. 路径常量 (基于本文件所在目录, 方便独立运行)
# ============================================================
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))     # 当前文件所在目录
PARENT_DIR = os.path.dirname(THIS_DIR)                    # 上一级目录 (即 day04/)
DATA_DIR = os.path.join(PARENT_DIR, 'data')               # ../data
MODEL_DIR = os.path.join(PARENT_DIR, 'model')             # ../model
IMG_DIR = os.path.join(PARENT_DIR, 'img')                 # ../img

# 数据文件路径
DATA_FILE = os.path.join(DATA_DIR, 'name_classfication.txt')

# 确保目录存在 (训练 / 绘图阶段要写模型文件 / 写图片)
for d in (DATA_DIR, MODEL_DIR, IMG_DIR):
    os.makedirs(d, exist_ok=True)


# ============================================================
# 5. 自检 (本文件被直接 python 执行时才会跑, 被 import 时不会跑)
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('公共常量自检:')
    print(f'常用字符(前30): {all_letters[:30]}...')
    print(f'常用字符数量: {n_letters}')                       # 期望: 57
    print(f'国家名: {categories}')
    print(f'国家名种类数: {category_num}')                    # 期望: 18
    print(f'数据文件路径: {DATA_FILE}')
    print(f'模型目录:   {MODEL_DIR}')
    print(f'图片目录:   {IMG_DIR}')
    print('=' * 60)
