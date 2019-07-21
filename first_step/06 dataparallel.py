import torch

# 在这个教程里，我们将学习如何使用数据并行（DataParallel）来使用多GPU。
#
# PyTorch非常容易的就可以使用GPU，可以用如下方式把一个模型放到GPU上：

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters 和 DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model： input size", input.size(),
              "output size", output.size())

        return output


#这是本教程的核心部分。首先，我们需要创建一个模型实例和检测我们是否有多个GPU。
# 如果我们有多个GPU，我们使用nn.DataParallel来包装我们的模型。
# 然后通过model.to(device)把模型放到GPU上。
model = Model(input_size, output_size)
if torch.cuda.device_count() >= 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)


for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside： input size", input.size(),
          "output_size", output.size())