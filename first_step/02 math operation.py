import torch

x = torch.rand(5,3)
y = torch.rand(5,3)
# 加法：形式一
print(x+y)
# 加法：形式二
print(torch.add(x, y))
# 加法：形式三 给定一个输出张量作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 加法：形式四 原位/原地操作（in-place）
y.add_(x)
print(y)

# 索引
print(y[:, 0])
# 改变形状：如果想改变形状，可以使用torch.view
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# 如果是仅包含一个元素的tensor，可以使用.item()来得到对应的python数值
x = torch.randn(1)
print(x)
print(x.item())

