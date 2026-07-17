"""
案例:
    演示 n-gram 特征

n-gram介绍:
    概述:
        就是连续的 n个词/字, 把这些连续片段当做1种特征(小词组特征), 帮我们分析文本规律.
    分类:
        uni-gram(1-gram): 把 每个词/字 拆出来
        bi-gram(2-gram):  找连续2个词的组合
        tri-gram(3-gram): 找连续3个词的组合
    目的:
        让那个计算机更好的理解 文本规律.
"""

# todo 1. 定义遍历, 记录: n的值, 一般 n-gram中的n取 2 或者 3, 这里以 2 举例
ngram_range = 2

# todo 2. 定义函数, 生成 n-gram 特征.
def create_ngram(input_list):
    # 1. 通过滑动窗口, 获取 n-gram特征.
    # i = 0 -> input_list[0:] -> [1, 3, 2, 1, 5, 3]
    # i = 1 -> input_list[1:] -> [3, 2, 1, 5, 3]
    sliced_lists = [input_list[i:] for i in range(ngram_range)]

    # 2. 使用zip()函数, 对切片列表进行组合.
    ngram_tuples = zip(*sliced_lists)

    # 3. 转换为集合(去重), 虽然本样例中无重复, 但是保证结果唯一.
    return set(ngram_tuples)


# todo 3. 测试代码
if __name__ == '__main__':
    # 1. 定义列表, 记录: 输入数值.
    input_list = [1, 3, 2, 1, 5, 3]

    # 2. 调用函数生成 n-gram 特征
    result = create_ngram(input_list)

    # 3. 输出结果.
    print(result)