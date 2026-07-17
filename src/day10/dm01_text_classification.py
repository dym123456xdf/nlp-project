"""
案例:
    演示 Fasttext实现 文本分类案例.

文本分类的几种场景:
    二分类问题:
    单标签多分类:
    多标签多分类:  一般会拆解成 多个 二分类问题解决.

例如: 数据格式如下, 如何处理呢?
            标签                                                      特征
    __label__soup __label__texture __label__standards       What is the correct consistency of a cream soup?

思路:
    转成 多个 二分类问题即可, 例如:

                    特征                                                标签
        What is the correct consistency of a cream soup?            __label__soup
        What is the correct consistency of a cream soup?            __label__texture
        What is the correct consistency of a cream soup?            __label__standards


细节:
    1. Fasttext确实训练和预测较快, 但是精度相对不是特别高, 我们可以优化下.
    2. 优化思路为(参考):
        1. 对数据做预处理, 统一小写, 单词和符号分离...
        2. 修改训练轮数.
        3. 调整学习率.
        4. 增加N-Gram
        5. 调整损失函数.
        6. 自动调参.
        7. 生产环境中, 可能会涉及到 多分类多标签的问题, 一般会拆解成: 多个二分类来解决.
        8. 保存和加载模型.

"""

# 导包
import fasttext


# todo 1. 直接训练.
def dm01_basic():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking_train.txt')

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('How is mass egg-frying performed?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('Using liquid nitrogen for tenderizing octopus?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking_valid.txt')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 2. 对数据做预处理 -> 统一小写, 符号和单词分离...
def dm02_data_preprocess():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking.pre.train')

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 3. 上步基础上, 调整: 训练轮数(默认: 5轮)
def dm03_epoch():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30)

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 4. 上步基础上, 调整: 学习率(默认: 5轮)
def dm04_learning_rate():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=1)

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 5. 上步基础上, 调整: N-Gram
def dm05_ngram():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=1, wordNgrams=2)

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 6. 上步基础上, 调整: 损失函数, 从传统的softmax() -> hs(层次softmax): 负采样 + softmax + 哈夫曼树
def dm06_loss():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=1, wordNgrams=2, loss='hs')

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')


    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')       # [验证集样本数量, 精度(准确率), 召回率]


# todo 7. 上步基础上, 调整: 从手动调参改为自动调参.
def dm07_auto():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised(
        input='./data/cooking.pre.train',                       # 指定 训练集的 路径.
        autotuneValidationFile='./data/cooking.pre.valid',      # 指定 验证集的 路径.
        autotuneDuration=180,                                   # 自动调参的时长限制, 默认: 5分钟, 这里改为: 3分钟.
    )

    # 2. 使用模型进行预测.
    # 预测第1条样本.
    result1 = model.predict('how is mass egg-frying performed ?')
    print(f'result1的预测结果为: {result1}')

    # 预测第2条样本
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?')
    print(f'result2的预测结果为: {result2}')

    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')  # [验证集样本数量, 精度(准确率), 召回率]


# todo 8. 在生产环境中, 可能会涉及到 多分类多标签的问题, 解决方案如下.
def dm08_multi_label():
    # 1. 获取训练模型(有监督训练: 有特征, 有标签) -> 得到训练好的模型.
    model = fasttext.train_supervised(
        input='./data/cooking.pre.train',  # 指定 训练集的 路径.
        epoch=25,                          # 训练轮数.
        lr=0.2,                            # 学习率.
        wordNgrams=2,                      # n-gram.
        loss='ova'                         # 多分类多标签问题: 逻辑回归 -> one-vs-all
    )

    # 2. 使用模型进行预测.
    # 参1: 预测第1条样本,  参2: k代表预测的标签数量,  参3: 阈值(至于大于该阈值的标签才会被保留)
    result1 = model.predict('how is mass egg-frying performed ?', k=3, threshold=0.5)
    print(f'result1的预测结果为: {result1}')

    # 参1: 预测第2条样本, 参2: k=-1的含义是: 预测所有标签(尽可能多的显示)
    result2 = model.predict('using liquid nitrogen for tenderizing octopus ?', k=-1)
    print(f'result2的预测结果为: {result2}')

    # 3. 测试模型(看看日志)
    result3 = model.test('./data/cooking.pre.valid')
    print(f'模型在测试集上的测试结果: {result3}')  # [验证集样本数量, 精度(准确率), 召回率]


# todo 9. 保存模型.
def dm09_save():
    # 1.获取训练模型(有监督训练: 有特征, 有标签) -> 获取训练好的模型.
    # 参1: 训练集的路径, 参2: 训练轮数, 参3: 学习率, 参4: n-gram.
    model = fasttext.train_supervised('./data/cooking.pre.train', epoch=30, lr=1, wordNgrams=2)

    # 2. 保存模型.
    model.save_model('./model/fasttext_model.model')

    # 3. 模型加载.
    model2 = fasttext.load_model('./model/fasttext_model.model')

    # 4.使用模型进行预测.
    # 参1: 预测第1条样本,  参2: k代表预测的标签数量,  参3: 阈值(至于大于该阈值的标签才会被保留)
    result1 = model2.predict('how is mass egg-frying performed ?', k=3, threshold=0.5)
    print(f'result1的预测结果为: {result1}')



# todo n.测试代码
if __name__ == '__main__':
    # 1. 测试: 直接训练.
    # dm01_basic()            # (3000, 0.14566666666666667, 0.06299553120945654)

    # 2. 测试: 对数据做预处理.
    # dm02_data_preprocess()  # (3000, 0.178, 0.07697852097448465)

    # 3. 测试: 增加训练轮数.
    # dm03_epoch()            # (3000, 0.544, 0.23526019893325645)

    # 4. 测试: 改变学习率.
    # dm04_learning_rate()    # (3000, 0.5873333333333334, 0.25400028830906735)

    # 5. 测试: 改变N-Gram.
    # dm05_ngram()            # (3000, 0.6166666666666667, 0.26668588727115466)

    # 6. 测试: 改变损失函数.
    # dm06_loss()             # (3000, 0.6023333333333334, 0.2604872423237711)

    # 7. 测试: 自动调参.
    # dm07_auto()               # (3000, 0.544, 0.23526019893325645)

    # 8. 测试: 多分类多标签问题.
    # dm08_multi_label()

    # 9. 测试: 保存模型.
    dm09_save()                 # (('__label__eggs',), array([0.73298752]))