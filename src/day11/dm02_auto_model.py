"""
案例:
    通过自动模型的方式 完成NLP常见的6个需求.
"""

import torch            # Pytorch深度学习框架, 张量计算, 自动求导...

# AutoConfig: 自动配置模型参数.
# AutoModel: 自动加载模型(即: 通用的模型加载类)
# AutoTokenizer: 自动加载和模型匹配的分词器(即: 通用的分词器加载类)
from transformers import AutoConfig, AutoModel, AutoTokenizer

# AutoModelForSequenceClassification: 用于 文本分类的 模型.
# AutoModelForMaskedLM: 用于 掩码语言建模的 模型.
# AutoModelForQuestionAnswering: 用于 问题回答的 模型.
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForQuestionAnswering

# AutoModelForSeq2SeqLM  用于 序列到序列的 建模(例如: 文本摘要)
# AutoModelForTokenClassification 用于 做token级分类(例如: NER) 的建模
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification

# from rich import print          # 输出美化(效果不明显, 你写不写都行)


# todo 1.情感分类任务.
# 例如: 情绪分类 -> '今天开心极了...'       积极的情绪
# 例如: 商品分类 -> '手机壳', '数码配件'...  '连衣裙' -> 服饰
def dm01_text_classification():
    # 1. 加载 Tokenizer -> 分词器对象, 把文本 转成 模型可接收的输入格式.
    my_tokenizer = AutoTokenizer.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\chinese_sentiment')

    # 2. 加载模型.
    my_model = AutoModelForSequenceClassification.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\chinese_sentiment')

    # 3. 文本转张量.
    # 3.1 定义文本.
    message = '人生该如何起头'

    # 3.2 思路1: return_tensors='pt' 返回的是: 二维张量(Tensor)
    msg_tensor1 = my_tokenizer.encode(
        text=message,               # 待编码的文本
        return_tensors='pt',        # 返回张量的类型, 默认为: None, 可选:'pt','tf','np'
        padding='max_length',       # 是否填充
        truncation=True,            # 是否截断
        max_length=5,               # (指定输入的)最大长度
    )
    # 格式为: msg_tensor1: tensor([[ 101,  782, 4495, 6421,  102]]), 类型: <class 'torch.Tensor'>
    # print(f'msg_tensor1: {msg_tensor1}, 类型: {type(msg_tensor1)}')

    # 3.3 思路2: 不用return_tensors='pt', 返回的是 一维列表, 需要手动转成张量.
    msg_list2 =my_tokenizer.encode(
        text=message,               # 待编码的文本
        padding='max_length',       # 是否填充
        truncation=True,            # 是否截断
        max_length=5,               # (指定输入的)最大长度
    )
    # 格式为: msg_list2: [101, 782, 4495, 6421, 102], 类型: <class 'list'>
    # print(f'msg_list2: {msg_list2}, 类型: {type(msg_list2)}')

    # 手动把上述的列表转成二维张量.
    msg_tensor2 = torch.tensor([msg_list2])
    # print(f'msg_tensor2: {msg_tensor2}, 类型: {type(msg_tensor2)}')

    # 4. 把数据送给模型进行推理.
    # 4.1 设置模型为: 推理模式(评估模式) -> 会关闭Dropout等训练特有的层.
    my_model.eval()
    # 4.2 直接调用模型(模型推理)
    result = my_model(msg_tensor2)
    print(f'result(包含模型输出的字典): {result}')

    # 4.3 使用 return_dict=False, 可以返回: 元组.
    result2 = my_model(msg_tensor2, return_dict=False)
    print(f'result2(包含模型输出的元组): {result2}')


# todo 2. 特征提取任务.
def dm02_feature_extraction():
    # 1. 加载 Tokenizer -> 分词器对象, 把文本 转成 模型可接收的输入格式.
    my_tokenizer = AutoTokenizer.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\bert-base-chinese')

    # 2. 加载模型.
    my_model = AutoModel.from_pretrained(r'C:\Software\CentOS_All\PretrainedModel\bert-base-chinese')

    # 3. 文本转张量.
    # 3.1 定义遍历, 记录: 待处理的文本.
    messsage = ['你是谁', '人生该如何起头']

    # 3.2 把上述的文本 转成 模型可接收的输入格式.
    # 返回值是字典形式, 其中各个键(key)的意思是:
    #  input_ids: 文本的id序列 -> 对应模型的 vocab.txt 文件
    #  token_type_ids: 文本的token序列
    #  attention_mask: 注意力掩码, 1(真实), 0(填充)
    msg_tensor = my_tokenizer.encode_plus(
        text = messsage,        # 待处理的文本
        return_tensors='pt',    # 返回张量的类型
        padding='max_length',   # 是否填充
        truncation=True,        # 是否截断
        max_length=30,          # (指定输入的)最大长度
    )
    # print(f'msg_tensor: {msg_tensor}')

    # 4. 给模型喂数据, 提取特征.
    my_model.eval()     # 切换到: 评估模式(推理模式)
    output = my_model(**msg_tensor)

    # 5. 获取模型输出 -> 打印结果.
    # print(f'output: \n{output}')
    # 最后一个(隐藏状态)层输出 -> [batch_size, seq_len, hidden_size]       # [1, 30, 768]
    print(f'output.last_hidden_state: {output.last_hidden_state.shape}')

    # 池化层输出 -> [batch_size, hidden_size], [1, 768]
    print(f'output.pooler_output: {output.pooler_output.shape}')


# todo 3. 完形填空任务.
def dm03_fill_mask():
    # 0. 定义遍历, 记录: 模型名.
    model_name = r'C:\Software\CentOS_All\PretrainedModel\chinese-bert-wwm'

    # 1. 加载Tokenizer分词器.
    my_tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 加载模型.
    my_model = AutoModelForMaskedLM.from_pretrained(model_name)

    # 3. 文本转张量.
    input = my_tokenizer.encode_plus('我想明天去[MASK]家吃饭', return_tensors='pt')
    # print(f'input: {input}')

    # 4. 给模型喂数据, 获取模型输出.
    my_model.eval()     # 评估模式
    output = my_model(**input)

    # 5. 获取结果, 并打印.
    # print(f'output: \n{output}')
    # print(f'output.logits: {output.logits.shape}')      # [1, 11, 21128]

    # [MASK]位置的索引: [1, 11, 21128]  -> [11, 21128] -> [21128]
    mask_predict_idx = torch.argmax(output.logits[0][6]).item()
    print(f'[MASK]位置的索引: {mask_predict_idx}')

    # 打印概率最高的字 -> '你'
    print(f'概率最高的字的: {my_tokenizer.convert_ids_to_tokens(mask_predict_idx)}')


# todo 4. 阅读理解 -> 抽取式问答任务, 即: 给定一段文本和问题, 让模型输出结果.
def dm04_question_answering():
    # 0. 定义遍历, 记录: 模型名
    model_name = r'C:\Software\CentOS_All\PretrainedModel\chinese_pretrain_mrc_roberta_wwm_ext_large'

    # 1. 加载分词器对象Tokenizer
    my_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 2. 加载模型.
    my_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # 3. 准备文本. 细节: 上下文最好不要写空格, 会影响模型定位, 空格会被删除, 可以换成标签符号.
    context = '我叫张三 我是一个程序员 我的喜好是乒乓球'
    questions = ['我是谁?', '我是做什么的?', '我的爱好是什么?']

    # 4. 该模型是针对于每个问题进行问答推理的, 即: 逐个问题生成, 此处要用for循环.
    for question in questions:
        # 4.1 把上述的问题 转成 模型可接收的输入格式.
        # 参1: 当前的问题. 参2: 包含答案的上下文文本. 参3: 返回张量的形式.
        input = my_tokenizer.encode_plus(question, context, return_tensors='pt')
        # 返回值字典, 各个键(key)的意思是:
        #  input_ids: [CLS] + 问题 + [SEP] + 上下文 + [SEP]     注意: 空格不统计.
        #  token_type_ids: 0 -> 问题, 1 -> 上下文
        #  attention_mask: 1 -> 真实, 0 -> 填充
        print(f'input: {input}')

        # 4.2 给模型喂数据, 获取模型输出.
        my_model.eval()
        output = my_model(**input)

        # 4.3 获取结果
        # print(f'output: \n{output}')
        print(f'output: {output.start_logits.shape}')       # [1, 26], 即: [CLS] + 问题 + [SEP] + 上下文 + [SEP]   表示: 答案的开头位置索引.
        print(f'output: {output.end_logits.shape}')         # [1, 26], 即: [CLS] + 问题 + [SEP] + 上下文 + [SEP]   表示: 答案的结束位置索引.
        # print('\n' * 3)

        # 4.4 计算模型预测的答案(开头的位置索引, 结束的位置索引)
        start, end = torch.argmax(output.start_logits), torch.argmax(output.end_logits) + 1     # 因为切片时, 包左不包右, 所以: 结束位置索引 + 1
        print(f'start: { start}, end: {end}')

        # 4.5 获取模型预测的答案.
        answer = my_tokenizer.convert_ids_to_tokens(input.input_ids[0][start:end])

        # 4.6 打印最终效果.
        print(f'question: {question}, answer: {answer}')
        print('\n' * 3)


# todo 5. 文本摘要: 摘要生成任务的输入一段文本, 输出是有一段概况, 简单的文字.
def dm05_summarization():
    # 1. 定义输入文本 -> 一段关于BERT模型预训练方法的加英文介绍.
    text = "BERT is a transformers model pretrained on a large corpus of English data " \
           "in a self-supervised fashion. This means it was pretrained on the raw texts " \
           "only, with no humans labelling them in any way (which is why it can use lots " \
           "of publicly available data) with an automatic process to generate inputs and " \
           "labels from those texts. More precisely, it was pretrained with two objectives:Masked " \
           "language modeling (MLM): taking a sentence, the model randomly masks 15% of the " \
           "words in the input then run the entire masked sentence through the model and has " \
           "to predict the masked words. This is different from traditional recurrent neural " \
           "networks (RNNs) that usually see the words one after the other, or from autoregressive " \
           "models like GPT which internally mask the future tokens. It allows the model to learn " \
           "a bidirectional representation of the sentence.Next sentence prediction (NSP): the models" \
           " concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to " \
           "sentences that were next to each other in the original text, sometimes not. The model then " \
           "has to predict if the two sentences were following each other or not."

    # 2. 定义遍历, 记录: 模型名
    model_name = r'C:\Software\CentOS_All\PretrainedModel\distilbart-cnn-12-6'
    # 3. 加载分词器对象Tokenizer 和 模型对象.
    my_tokenizer = AutoTokenizer.from_pretrained(model_name)
    my_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 4. 将输入文本 转成 模型可接收的输入格式.
    input = my_tokenizer(text, return_tensors='pt')
    print(f'input: {input}')

    # 5. 把输入文本喂给模型, 获取预测结果.
    # 5.1 切换模型模式为评估模式.
    my_model.eval()
    # 5.2 模型预测, 文本摘要
    output = my_model.generate(input['input_ids'])
    print(f'output: {output}')

    # 6. 打印预测结果, 即: 把文本只要 -> 可读问题
    # 扩展: 只能将ids还原为token, 如果感兴趣, 你可以看一BPE(Byte-Pair Encoding), 字节对编码(切词), GPT底层就是用它(BPE)切词的.
    print(my_tokenizer.convert_ids_to_tokens(output[0]))

    # decode()解码
    # skip_special_tokens: 跳过特殊标记 -> 即: 去除4token前后的特殊字符.
    # clean_up_tokenization_spaces: 是否保留原始分词的空格.
    print([my_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output])
    print([my_tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in output])


# todo 6. NER任务(Named Entity Recognition) 命名实体识别.
# 命名实体识别: 本质上是一个分类任务(又叫序列标注任务), 实体词识别是句法分析的基础, 而句法分析又是NLP任务的核心.
def dm06_ner():
    # 0. 定义遍历, 记录: 模型名.
    model_name = r'C:\Software\CentOS_All\PretrainedModel\roberta-base-finetuned-cluener2020-chinese'

    # 1. 加载分词器.
    my_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 2. 加载模型.
    my_model = AutoModelForTokenClassification.from_pretrained(model_name)
    # 3. 加载配置文件.
    config = AutoConfig.from_pretrained(model_name)

    # 4. 数据张量化 -> 把输入文本转换成模型可接收的输入格式.
    input = my_tokenizer.encode_plus('我爱北京天安门, 天安门上太阳升!', return_tensors='pt')
    print(f'input: {input}')

    # 5. 把数据喂给模型, 获取结果.
    my_model.eval()
    logits = my_model(input.input_ids).logits
    print(f'logits: {logits.shape}')        # [1, 18, 32]

    # 6. 处理预处理结果, 并显示.
    # 6.1 把输入id 转回 token形式(用于格式化)
    # 格式为: ['[CLS]', '我', '爱', '北', '京', '天', '安', '门', ',', '天', '安', '门', '上', '太', '阳', '升', '!', '[SEP]']
    input_tokens = my_tokenizer.convert_ids_to_tokens(input.input_ids[0])
    print(f'input_tokens: {input_tokens}')

    # 6.2 用zip()函数, 把 token 和 预测的标签概率分布组合到一起.
    outputs = []        # 格式为: [('我', 'O'), ('爱', 'O'), ('北京', 'LOC')...]
    # 遍历每个token(18个) 和 预测的概率标签分布:  [18, 32]
    for token, value in zip(input_tokens, logits[0]):
        # 过滤掉特殊字符.
        if token in my_tokenizer.all_special_tokens:
            continue
        # 走到这里, 说明不是特殊字符, 就获取: 32个标签的概率分布中, 最大概率标签的索引.
        idx = torch.argmax(value).item()
        # print(token, idx)

        # 构建元组, 即: (token, 标签)
        outputs.append((token, config.id2label[idx]))

    # 7. 最终输出结果, 格式为: [('我', 'O'), ('爱', 'O'), ('北京', 'LOC')...]
    # 格式为: [('我', 'O'), ('爱', 'O'), ('北', 'B-address'), ('京', 'I-address'), ('天', 'I-address'), ('安', 'I-address'), ('门', 'I-address'), (',', 'O'), ('天', 'B-address'), ('安', 'I-address'), ('门', 'I-address'), ('上', 'O'), ('太', 'O'), ('阳', 'O'), ('升', 'O'), ('!', 'O')]
    print(f'outputs: {outputs}')





# todo 7.测试代码
if __name__ == '__main__':
    # 1. 测试Transformers的自动模型方式处理NLP任务 -> 情感分类
    # dm01_text_classification()

    # 2. 测试Transformers的自动模型方式处理NLP任务 -> 特征提取
    # dm02_feature_extraction()

    # 3. 测试Transformers的自动模型方式处理NLP任务 -> 完形填空
    # dm03_fill_mask()

    # 4. 测试Transformers的自动模型方式处理NLP任务 -> 阅读理解
    # dm04_question_answering()

    # 5. 测试Transformers的自动模型方式处理NLP任务 -> 文本摘要
    # dm05_summarization()

    # 6. 测试Transformers的自动模型方式处理NLP任务 -> NER
    dm06_ner()