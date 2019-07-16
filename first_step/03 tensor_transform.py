import torch
# NumPy桥
# 将一个Torch张量转换为一个NumPy数组是轻而易举的事情，反之亦然。
# Torch张量和NumPy数组将共享它们的底层内存位置，更改一个将更改另一个。

# 将torch的Tensor转化为NumPy数组
a = torch.ones(5)
print(a) # tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b) # [1. 1. 1. 1. 1.]
# 数值的改变
a.add_(1)
print(b) # [2. 2. 2. 2. 2.]

# 将NumPy数组转化为Torch张量
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)

np.add(a,1,out=a)
print(b)
a = a+1
print(a)
print(b)

x = torch.rand(2,2)
y = torch.rand(2,2)
# cuda tensor
# 张量可以使用.to方法移动到任何设备（device）上：
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    print(device)
    y = torch.ones_like(y,device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype

    # tensor([[1.4342, 1.5242],
    #         [1.4269, 1.3429]], device='cuda:0')
    # tensor([[1.4342, 1.5242],
    #         [1.4269, 1.3429]], dtype=torch.float64)