# https://www.w3cschool.cn/pytorch/pytorch-w18e3be1.html

# 训练一个图片分类器
# 我们将按顺序做以下步骤：
#
# 通过torchvision加载CIFAR10里面的训练和测试数据集，并对数据进行标准化
# 定义卷积神经网络
# 定义损失函数
# 利用训练数据训练网络
# 利用测试数据测试网络
# 1.加载并标准化CIFAR10
# 使用torchvision加载 CIFAR10 超级简单。

import torch
import torchvision
import torchvision.transforms as transforms
# torchvision 数据集加载完后的输出是范围在 [ 0, 1 ] 之间的 PILImage。我们将其标准化为范围在 [ -1, 1 ] 之间的张量。

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# ## 输出图像的函数
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# ## 随机获取训练图片
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
#
# ## 显示图片
# imshow(torchvision.utils.make_grid(images))
# ## 打印图片标签
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 2.定义一个卷积神经网络
# 将之前神经网络章节定义的神经网络拿过来，并将其修改成输入为3通道图像(替代原来定义的单通道图像）。

import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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




net = Net()
# 3.定义损失函数和优化器
# 我们使用多分类的交叉熵损失函数和随机梯度下降优化器(使用 momentum ）。

import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 4.训练网络
# 事情开始变得有趣了。我们只需要遍历我们的数据迭代器，并将输入“喂”给网络和优化函数。

for epoch in range(2):  # loop over the dataset multiple times


    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data


        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


print('Finished Training')