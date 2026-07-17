"""
案例:
    演示下 masked_fill()函数的用法 -> 掩码操作.
"""

# # 导包
# import torch
#
# # 1. 生成1个随机整数张量, 取值范围: [1, 10), shape: [3, 5]
# input = torch.randint(1, 10, (3, 5))
# print(f'input: \n{input}, shape: {input.shape}')
#
#
# # 2. 定义掩码张量mask
# mask = torch.tensor([
#     [1, 2, 3, 0, 0],
#     [2, 3, 4, 0, 0],
#     [2, 3, 4, 1, 0]
# ])
#
# # 3. 处理mask, 把非0的值都改成1
# mask[mask != 0] = 1
# print(f'mask: \n{mask}, shape: {mask.shape}')
#
# # 4. 用 masked_fill()函数处理input, 进行掩码操作.
# # 参1: 要被处理的张量
# # 参2: mask == 0: 布尔条件, 找到mask中值为0的位置.
# # 参3: 把input中对应mask为0位置的元素, 替换成 -1e9(一个非常小的数, 常用于注意力机制的遮挡)
# result = torch.masked_fill(input, mask == 0, -1e9)
# # result = torch.masked_fill(input, mask == 0, 1e-9)
#
# # 5. 打印结果.
# print(f'result: \n{result}, shape: {result.shape}')


assert 10 // 2 == 5
print('看看我')