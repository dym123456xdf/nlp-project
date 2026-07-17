# 这个.py脚本文件中, 放的都是 某些函数的 实例Demo, 简单看看即可.


# 需求1: 演示下map函数的用法.
# todo 1. 定义1个函数, 接收1个参数, 并返回其 加2的结果.
def fun(x):
    return x + 2

# todo 2. 定义函数, 演示map()函数的用法, 即: 把上述的 fun()函数, 作用到列表中的每个元素熵.
def dm01_map():
    # 1. 将上述的 fun()函数, 作用到: 列表中的每个元素上.
    # map(): 返回1个迭代器对象, 此时并未实际执行运算.
    result = map(fun, [1, 2, 3, 4, 5])
    print(f'result: {result}')      # <map object at 0x0000015448175510>

    # 2. 第一次遍历map对象.
    for i in result:
        print(i)                    # 依次输出: 3, 4, 5, 6, 7
    print(' -.- ' * 10)

    # 3. 第二次尝试遍历map对象.
    print(f'result: {list(result)}')    # []
    print(' -.- ' * 10)

    # 4. 使用 lambda表达式实现 相同功能.
    result2 = list(map(lambda x: x + 2, [10, 20, 30]))
    print(f'result2: {result2}')




# 需求2: 演示 chain()函数的用法.
# 导包
from itertools import chain
import jieba

# 定义函数, 测试: chain()函数的用法.
def dm02_chain():
    # 1. 定义2个列表.
    list1, list2 = [1, 2, 3], [2, 3, 4]

    # 2. chain(): 它是"惰性"的, 即: 创建chain()对象时, 不会立即遍历底层的可迭代对象, 只有在实际迭代(例如: 转列表, 遍历打印等)时, 才会逐个获取元素, 节省内存.
    # 一旦迭代完毕, 再次迭代同一个chain对象, 不会得到数据, 因为: 迭代器的元素已经"耗尽"
    result = chain(list1, list2)
    print(f'result: {result}')              # <itertools.chain object at 0x000001FACDC8BDF0>
    print(f'result: {list(result)}')        # [1, 2, 3, 2, 3, 4]

    # chain()函数, 有点类似于: list1.extend(list2)
    # list1.extend(list2)
    # print(f'result2: {list1}')


    # # 3. 重新定义列表, 记录两个句子.
    # list1 = ['今天天很好', '今天天很热']
    # # 4. 对上述的两个句子进行切词.
    # tmp = map(lambda x: jieba.lcut(x), list1)
    # # print(f'tmp: {tmp}')            # <map object at 0x0000020E0E0D0E50>
    # # print(f'tmp: {list(tmp)}')      # [['今天', '天', '很', '好'], ['今天', '天', '很', '热']]
    #
    # # 5. 用 chain()函数, 链接两个列表.  这里的 * 意思是: 告诉函数: 把后面的参数, 逐个取出来, 拼接到前面.
    # # result = list(chain(*tmp))        # 不会去重
    # result = set(chain(*tmp))           # 会去重
    # print(f'result: {result}')


    # 6. 合并版.
    list1 = ['今天天很好', '今天天很热']
    result = set(chain(*map(lambda x: jieba.lcut(x), list1)))
    print(f'result: {result}')




# 需求3: 演示 zip()函数入门, 功能: 合并迭代对象.
def dm03_zip():
    # 1. 定义两个列表
    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [2, 3, 4]

    # 2. 使用zip()函数, 合并: 迭代对象.
    result = zip(list1, list2)
    print(f'result: {result}')      # <zip object at 0x000001FACDC8BDF0>

    # 3. 把上述的zip对象, 转成列表, 查看内容.
    print(list(result))             # [(1, 2), (2, 3), (3, 4)]  以短的为主.


    # 4. 演示: zip的 解包(*) 的用法.
    # 4.1 定义嵌套列表(二维结构)
    list3 = [
        [1, 2, 3],
        [3, 4, 5]
    ]

    # 4.2 不使用解包的zip()调用, 会把list3整体当做单个迭代对象处理.
    print(list(zip(list3)))     # [([1, 2, 3], ), ([3, 4, 5], )]

    # 4.3 使用解包的zip()调用, 会把list3中的元素, 逐个取出.
    print(list(zip(*list3)))    # [(1, 3), (2, 4), (3, 5)]



# todo n. 测试代码
if __name__ == '__main__':
    # 1. 测试: map() 函数的用法
    # dm01_map()

    # 2. 测试: chain() 函数的用法
    # dm02_chain()

    # 3. 测试: zip() 函数的用法
    dm03_zip()

