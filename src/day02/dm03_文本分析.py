"""
案例:
    演示 文本分析的常见操作.

文本分析的作用:
    帮助我们理解数据语料, 快速检查出语料中可能存在的问题.
    例如:
        数据质量类:      错别字, 语法错误, 重复内容, 缺失值, 噪声数据...
        分布不均衡问题:   标签分布不均, 句子长度不同...

文本分析的方式:
    1. 标签的数量分布
    2. 句子长度分布
    3. 词频统计和关键字词云
"""

# 导包
import jieba                        # 分词包
import seaborn as sns               # 画图包
import pandas as pd                 # 数据处理包
import matplotlib.pyplot as plt     # 画图包, 推荐 pip install matplotlib==3.9
from itertools import chain         # 迭代器工具
import jieba.posseg as pseg         # 词性标注包, 例如: 动词, 名词, 形容词...
from wordcloud import WordCloud     # 词云包

# 解决中文乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']        # 如果是Mac本, 可以换成: 'Arial Unicode  MS'
plt.rcParams['axes.unicode_minus'] = False


# todo 1. 定义函数, 实现: 训练集 和 测试集的 标签分布的  可视化统计.
def dm01_label_sns_countplot():
    # 1. 设置538风格 -> 一种具有现代感的可视化风格(不做也行)
    plt.style.use('fivethirtyeight')

    # 2. 读取训练集 和 测试集
    # 参1: 文件路径.   参2: 列分隔符(csv文件用,  tsv文件用\t)
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')
    # print(train_data.head())

    # 3. 统计训练集标签的 0(负样本) 和 1(正样本) 的数量, 并可视化, 采用: 计数柱状图.
    # 参1: x轴标签,  参2: 数据集,  参3: 用于分组的分类变量,  参4: 是否显示图例(默认为True)
    sns.countplot(x='label', data=train_data, hue='label', legend=False)
    plt.title('train_label')    # 设置标题
    plt.tight_layout()          # 紧凑布局
    plt.show()

    # 4. 统计测试集标签的 0(负样本) 和 1(正样本) 的数量, 并可视化, 采用: 计数柱状图.
    sns.countplot(x='label', data=dev_data, hue='label', legend=False)
    plt.title('dev_label')
    plt.tight_layout()
    plt.show()


# todo 2. 定义函数, 实现: 训练集 和 测试集的 句子长度分布的 可视化统计.
def dm02_len_sns_distplot():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 计算训练集的 每个句子的 长度.
    # 思路1: map()函数的方式实现.
    # train_data['sentence_length'] = list(map(函数对象的功能是 获取句子长度, 句子))
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 思路2: apply()函数的方式实现.
    # train_data['sentence_length'] = train_data['sentence'].apply(lambda x: len(x))
    # print(train_data.head())

    # 3. 绘制训练集的 句子长度分布.
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=train_data)
    plt.title('训练集句子长度分布_计数柱状图')
    plt.xticks([])      # 隐藏x轴刻度值
    plt.show()

    # 图2: 密度曲线图.
    # 旧版写法
    # sns.displot(x='sentence_length', data=train_data, kde=True)
    # plt.yticks([])
    # plt.show()

    # histplot() 直方图
    sns.histplot(x='sentence_length', data=train_data, kde=True)
    plt.title('训练集句子长度分布_密度曲线图')
    plt.show()


    # 4. 计算测试集的 每个句子的 长度.
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))
    # 图1: 计数柱状图
    sns.countplot(x='sentence_length', data=dev_data)
    plt.title('测试集句子长度分布_计数柱状图')
    plt.xticks([])
    plt.show()

    # 图2: 密度曲线图.
    sns.histplot(x='sentence_length', data=dev_data, kde=True)
    plt.title('测试集句子长度分布_密度曲线图')
    plt.show()


# todo 3. 定义函数, 实现: 训练集 和 测试集的 正负样本长度散点分布.
def dm03_sns_stripplot():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 获取(训练集)数据长度列
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))

    # 3. 获取(测试集)数据长度列
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 4. 统计正负样本长度的 散点分布
    # 训练集
    # 参1: x轴标签,  参2: y轴标签,  参3: 数据集,  参4: 用于分组的字段(即: 不同组颜色不同)
    sns.stripplot(x='label', y='sentence_length', data=train_data, hue='label')
    plt.title('训练集正负样本长度散点分布')
    plt.show()

    # 测试集
    sns.stripplot(x='label', y='sentence_length', data=dev_data)
    plt.title('测试集正负样本长度散点分布')
    plt.show()



# todo 4. 定义函数, 获取: 训练集和测试集的 词汇总数(去重后的).
def dm04_get_word_count():
    # 1. 读取训练集 和 测试集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')
    dev_data = pd.read_csv('./data/dev.tsv', sep='\t')

    # 2. 统计训练集的 词汇总数(去重后的).
    train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    print(f'训练集共包含不同词汇总数为: {len(train_vocab)}')


    # 3. 统计测试集的 词汇总数(去重后的).
    dev_vocab = set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))
    print(f'测试集共包含不同词汇总数为: {len(dev_vocab)}')



# todo 5. 定义函数, 实现: 训练集和测试集的 高频形容词词云.
# todo 5.1 定义函数, 获取文本中的形容词列表.
def get_a_list(text):
    # 1. 定义空列表, 用于存储: 文本中的形容词.
    a_list = []

    # 2. 使用jieba的词性标注功能, 切分文本, 并遍历获取到每个词.
    for value in pseg.lcut(text):   # text = '一个句子'
        # print(f'value: {value}')            # 词 词性
        # print(f'value.word: {value.word}')  # 词
        # print(f'value.flag: {value.flag}')  # 词性
        # 3. 判断词性, 是否是形容词, 如果是, 就添加到列表中.
        if value.flag == 'a':
            a_list.append(value.word)

    # 4. 返回列表.
    return a_list


# todo 5.2 定义函数, 根据 词列表 产生 词云图.
def get_word_cloud(keywords_list):
    # 1. 实例化词云生成器, 设置字体路径, 最大显示数, 背景色...
    wordcloud = WordCloud(
        font_path='./data/SimHei.ttf',      # 字体路径
        max_words=100,                      # 最大显示数
        background_color='white'            # 背景色
    )

    # 2. 将关键词列表 -> 转换为 空格分割的字符串, 适配: 词云输入格式.
    keywords_str = ' '.join(keywords_list)
    # 3. 根据关键词字符串, 生成: 词云.
    wordcloud.generate(keywords_str)

    # 4. 配置并显示词云图像.
    # 4.1 创建新的绘图窗口.
    plt.figure()
    # 4.2 生成词云(绘制图像)
    # 参1: 生成词云图像的数据,  参2: 设置图像插值方法为: 双线性插值
    plt.imshow(wordcloud, interpolation='bilinear')
    # 4.3 隐藏坐标轴.
    plt.axis('off')
    # 4.4 显示图像.
    plt.show()


# todo 5.3 定义函数, 实现: 训练集和测试集的 高频形容词词云.
def dm05_word_cloud():
    # 场景1: 处理 训练集 -> 正样本
    # 1. 读取训练集
    train_data = pd.read_csv('./data/train.tsv', sep='\t')

    # 2. 处理 训练集的 正样本(label=1) -> 生成词云.
    # 2.1 筛选label=1的样本, 并提取句子列.
    p_train_data = train_data[train_data['label'] == 1]['sentence']
    # 2.2 对每个正样本句子, 提取形容词列表, 并合并为1个完整的 形容词列表.
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x),  p_train_data))
    # 2.3 调用词云函数, 根据形容词列表, 绘制词云.
    get_word_cloud(p_a_train_vocab)

    # 分隔符
    print(' =.= ' * 10)

    # 场景2: 处理 训练集 -> 负样本
    # 2. 处理 训练集的 负样本(label=0) -> 生成词云.
    # 2.1 筛选label=0的样本, 并提取句子列.
    p_train_data = train_data[train_data['label'] == 0]['sentence']
    # 2.2 对每个负样本句子, 提取形容词列表, 并合并为1个完整的 形容词列表.
    p_a_train_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
    # 2.3 调用词云函数, 根据形容词列表, 绘制词云.
    get_word_cloud(p_a_train_vocab)





# todo n.测试代码
if __name__ == '__main__':
    # 1. 测试: 训练集 和 测试集的 标签分布的  可视化统计.
    dm01_label_sns_countplot()

    # 2. 测试: 训练集 和 测试集的 句子长度分布的 可视化统计.
    # dm02_len_sns_distplot()

    # 3. 测试: 训练集 和 测试集的 正负样本长度散点分布.
    # dm03_sns_stripplot()

    # 4. 测试: 获取: 训练集和测试集的 词汇总数(去重后的).
    # dm04_get_word_count()

    # 5. 测试: 训练集和测试集的 高频形容词词云.
    # dm05_word_cloud()















