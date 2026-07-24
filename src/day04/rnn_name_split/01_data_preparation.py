"""
01_data_preparation.py
=======================

对应原文件 todo 1 / 2 / 3 / 4 / 5: 数据预处理完整流程.

流程:
    源文件 -> read_data() 函数 -> NameClassDataset 类 -> DataLoader 数据加载器

本文件拆出来后, 三个核心组件:
    1. read_data(file_path)        -> 把 txt 文件解析成 [人名列表, 国家列表]
    2. NameClassDataset            -> 自定义 Dataset, 把人名转 one-hot
    3. get_dataloader() / get_loader() -> 返回 DataLoader 供训练用
"""

# ============================================================
# 导入公共代码
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
    torch, Dataset, DataLoader,
    all_letters, n_letters, categories, DATA_FILE
)


# ============================================================
# todo 3. 读取源数据到内存
# ============================================================
def read_data(file_path):
    """
    读取源数据到内存中, 并把 特征(人名) 和 标签(国家) 分别存储到两个列表中.
    :param file_path: 源数据文件的路径
    :return: my_list_x: 存储的人名(特征), my_list_y: 存储的国家名(标签)
    """
    # 1. 创建两个列表, 分别存储: 人名(特征), 国家名(标签)
    my_list_x, my_list_y = [], []

    # 2. 关联文件, 并读取其内容(逐行读取)
    with open(file_path, 'r', encoding='utf-8') as f:
        # 3. 遍历, 获取到每一行的数据.
        for line in f.readlines():
            # 4. 过滤无效数据, 假设(整行的)长度 小于等于5, 就过滤掉.  整行长度 = 人名 + '\t' + 国家名
            if len(line) <= 5:
                continue
            # 5. 走到这里, 说明该行数据是有效数据(即: 行长度 > 5), 处理后, 添加到对应的列表中.
            x, y = line.strip().split('\t')
            # 6. 添加到对应的列表中.
            my_list_x.append(x)
            my_list_y.append(y)

    # 扩展: 查看下数据集的长度, 即: 判断数据是否正确.
    print(f'my_list_x: {len(my_list_x)}')        # 期望: 20074
    print(f'my_list_y: {len(my_list_y)}')        # 期望: 20074

    # 7. 返回解析后的 样本 和 标签.
    return my_list_x, my_list_y


# ============================================================
# todo 4. 自定义 Dataset 类: 把每个人名转成 one-hot 编码
# ============================================================
class NameClassDataset(Dataset):
    """
    自定义数据集, 主要做两件事:
        1. 把 人名 字符串 转成 [seq_len, n_letters] 的 one-hot 张量
           例如 'Ding' (4 个字符) -> [4, 57] 的零一张量, 对应字母位置为 1.
        2. 把 国家 字符串 转成 在 categories 列表中的索引, 作为标签.
    """

    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x          # 存储样本数据列表
        self.my_list_y = my_list_y          # 存储标签数据列表
        self.sample_len = len(my_list_x)    # 计算样本总数并存储, 20074

    def __len__(self):
        """获取样本总数."""
        return self.sample_len

    def __getitem__(self, index):
        """
        根据指定的索引, 获取其对应的样本, 并进行 one-hot编码 和 张量转换.
        :param index: 样本索引
        :return: tensor_x: 人名(特征)的one-hot编码, tensor_y: 国家(标签)的张量表示
        """
        # 1. 索引边界校验
        index = min(max(index, 0), self.sample_len - 1)

        # 2. 按照索引获取原始样本 和 标签
        x = self.my_list_x[index]           # 例如 'Ding'
        y = self.my_list_y[index]           # 例如 'Chinese'

        # 3. 人名数据转换为 one-hot 编码
        # 3.1 生成全0张量, shape = [人名字符数, 字符表长度]
        tensor_x = torch.zeros(len(x), n_letters)       # 例如 [4, 57]
        # 3.2 遍历人名, 设置对应字符位置为 1
        for li, letter in enumerate(x):
            letter_index = all_letters.find(letter)     # 字母在全局字母表中的索引
            tensor_x[li][letter_index] = 1              # one-hot

        # 4. 国家数据转换为索引张量 (标签)
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)

        # 5. 返回结果
        return tensor_x, tensor_y


# ============================================================
# todo 5. 数据加载器 (Tensor -> TensorDataset -> DataLoader)
# ============================================================
def get_loader(batch_size=1, shuffle=True):
    """
    一站式封装: 读数据 -> 建 Dataset -> 建 DataLoader.

    :param batch_size: 批大小, 默认 1 (跟原文件保持一致, 适合教学)
    :param shuffle:    是否打乱, 训练集打乱, 测试集不打乱
    :return: DataLoader 对象, 可以直接 for batch in loader
    """
    # 1. 读取数据
    my_list_x, my_list_y = read_data(DATA_FILE)

    # 2. 创建数据集对象
    name_class_dataset = NameClassDataset(my_list_x, my_list_y)

    # 3. 创建数据加载器
    my_dataloader = DataLoader(
        name_class_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return my_dataloader


# ============================================================
# 自检: 单独运行此文件时, 打印一个样本看看 one-hot 编码长什么样
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('todo 3 验证: 读取数据')
    print('=' * 60)
    xs, ys = read_data(DATA_FILE)
    print(f'前 5 个样本(人名): {xs[:5]}')
    print(f'前 5 个标签(国家): {ys[:5]}')

    print('\n' + '=' * 60)
    print('todo 4 验证: Dataset 取单条样本, 看 one-hot 编码')
    print('=' * 60)
    ds = NameClassDataset(xs, ys)
    print(f'数据集总长度: {len(ds)}')                       # 20074

    tensor_x, tensor_y = ds[0]
    print(f'xs[0]: "{xs[0]}", 对应 ys[0]: "{ys[0]}"')
    print(f'tensor_x 形状: {tensor_x.shape}')              # torch.Size([len, 57])
    # 把内容打印, 看一眼前面几个 one-hot 行
    print(f'tensor_x 内容: \n{tensor_x}')
    print(f'tensor_y 形状: {tensor_y.shape}, 值: {tensor_y.item()}')

    print('\n' + '=' * 60)
    print('todo 5 验证: DataLoader 取一个 batch')
    print('=' * 60)
    loader = get_loader(batch_size=1, shuffle=True)
    for x, y in loader:
        print(f'x.shape: {x.shape}')      # torch.Size([1, seq_len, 57])
        print(f'y.shape: {y.shape}')      # torch.Size([1])
        # 这里只打印前两行, 看 one-hot 张量大概长什么样
        print(f'x[0] 前 2 行: \n{x[0][:2]}')
        print(f'y: {y.item()}')
        break

    print('\n数据预处理部分全部 OK!')
