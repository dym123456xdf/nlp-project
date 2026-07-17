
# 演示 view(1) 一维 和 view(1, -1) 二维


# 导包
import torch

# 1. 准备数据
y = torch.randint(0, 4345, size=(1, 7))
print(f'语法句子: {y}')     #  shape: (1, 7)


# 2. 打印结果
print(y[0])        # [ 876, 1233, 4015,   94, 3325, 4088, 2610]
print(y[0][3])              #  tensor(94)

print(y[0][3].view(1))      # tensor([94])
print(y[0][3].view(-1))     # tensor([94])
print(y[0][3].view(1, -1))  # tensor([[94]])