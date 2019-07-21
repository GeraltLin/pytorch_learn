import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy




# 什么是 状态字典?
# 在Pytorch中，torch.nn.Module 模型的可学习参数(即权重和偏差)包含在模型的 parameters 中，
# (使用model.parameters()可以进行访问)。
# state_dict 仅仅是python字典对象，它将每一层映射到其参数张量。
# 注意，只有具有可学习参数的层(如卷积层、线性层等)的模型才具有 state_dict 这一项。
# 优化目标 torch.optim 也有 state_dict 属性，它包含有关优化器的状态信息，以及使用的超参数。
# 因为 state_dict 的对象是python字典，所以他们可以很容易的保存、更新、更改和恢复，
# 为Pytorch模型和优化器添加了大量模块。


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# conv1.weight 	 torch.Size([6, 3, 5, 5])
# conv1.bias 	 torch.Size([6])
# conv2.weight 	 torch.Size([16, 6, 5, 5])
# conv2.bias 	 torch.Size([16])
# fc1.weight 	 torch.Size([120, 400])
# fc1.bias 	 torch.Size([120])
# fc2.weight 	 torch.Size([84, 120])
# fc2.bias 	 torch.Size([84])
# fc3.weight 	 torch.Size([10, 84])
# fc3.bias 	 torch.Size([10])

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# state 	 {}
# param_groups 	 [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [139620603375048, 139620603375192, 139620603375264, 139620603375120, 139620603375408, 139620603375336, 139620603375552, 139620603375480, 139620603523144, 139620603523216]}]


# 保存/加载 state_dict (推荐使用)
# 保存:
#
# torch.save(model.state_dict(), PATH)
# 加载:
#
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()


# 当保存好模型用来推断的时候，只需要保存模型学习到的参数，
# 使用 torch.save() 函数来保存模型 state_dict ,给模型恢复提供最大的灵活性。
# 在 Pytorch 中最常见的模型保存使用 ‘.pt’ 或者是 ‘.pth’ 作为模型文件扩展名。

# 请记住，在运行推理之前，
# 务必调用 model.eval() 去设置 dropout 和 batch normalization 层为评估模式。
# 如果不这么做，可能导致模型推断结果不一致。

# 请注意 load_state_dict() 函数只接受字典对象，而不是保存对象的路径。
# 这就意味着在你传给 load_state_dict() 函数之前，
# 你必须反序列化你保存的 state_dict。
# 例如，你无法通过 model.load_state_dict(PATH)来加载模型。


# torch.save(model, PATH)
# # Model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()
# 此部分保存/加载过程使用最直观的语法并涉及最少量的代码。
# 以Pythonpickle模块的方式来保存模型。
# 这种方法的缺点是序列化数据受限于某种特殊的类而且需要确切的字典结构。
# 这是因为pickle无法保存模型类本身。
# 相反，它保存包含类的文件的路径，该文件在加载时使用。
# 因此，当在其他项目使用或者重构之后，您的代码可能会以各种方式中断。