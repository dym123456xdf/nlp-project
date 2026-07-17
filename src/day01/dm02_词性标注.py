"""
案例:
    代码演示下 文本预处理之  词性标注(POS)

名词解释:
    NER:
        概述:
            命名实体识别, Named Entity Recognition, 就是识别出一段文本中可能存在的<<命名实体>>.
        命名实体:
            人名, 地名, 机构名等专有名词.
        作用:
            和词汇(分词)一样, 也是人类理解文本的基础单元.
        细节:
            后续我们NLP的第一个项目, 就是NER, 后续详解, 目前了解概念即可.

    POS:
        概述:
            词性标注, Part Of Speech Tagging, 语言中对词的一种分类方法.
            以 语法特征 为主要依据, 兼顾词汇意义对词进行划分.
        例如:
            名词, 动词, 形容词...
        大白话:
            POS(词性标注) = 标注出一段文本中 每个词汇的 词性.
"""

# 导包
import jieba.posseg as pseg

# 1. 定义遍历, 记录: 待分词并标注词性的 文本内容.
content = '我爱武汉黄鹤楼和东湖'

# 2. 使用 jieba进行 分词 和 词性标注
result = pseg.lcut(content)
print(f'result: {result}')

# 3. 获取每组数据, 即: 词 <-> 词性
for word, flag in result:
    print(f'词语: {word}, 词性: {flag}')