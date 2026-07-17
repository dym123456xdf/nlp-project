"""
案例:
    具体模型方式完成 NLP任务 -> 其方式 和 自动模型几乎一致.
    只是在加载模型 和 分词器的API上稍有不同, 所以这里我们只演示 1个 案例.

细节(记忆):
    除了第3方库不一样, 调用方式稍有不同, 其它都一模一样.
"""


# 导包
import torch            # Pytorch深度学习框架, 张量计算, 自动求导...

# BertTokenizer: 自动加载和模型匹配的分词器(即: 通用的分词器加载类)
# BertModelForMaskedLM: 用于 掩码语言建模的 模型.
from transformers import BertTokenizer,  BertForMaskedLM



# todo 1. 完形填空 -> 一次只能预测1个[MASK], 如果要多个[MASK], 则必须通过循环实现.
def dm01_fill_mask():
    # 0. 定义遍历, 记录: 模型名.
    model_name = r'C:\Software\CentOS_All\PretrainedModel\chinese-bert-wwm'

    # 1. 加载Tokenizer分词器.
    my_tokenizer = BertTokenizer.from_pretrained(model_name)

    # 2. 加载模型.
    my_model = BertForMaskedLM.from_pretrained(model_name)

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


# todo 2. 测试代码
if __name__ == '__main__':
    # 1. 具体模型方式 -> 演示: 完形填空.
    dm01_fill_mask()