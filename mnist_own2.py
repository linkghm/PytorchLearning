# -*- coding: utf-8 -*-
# @Time    : 2019/3/31 12:40
# @Author  : GUO Huimin
# @Email   : guohuimin2619@foxmail.com
# @FileName: mnist_own2.py

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Related parameters
Learning_rate = 0.01
Epoch = 3
Batch_size = 50


# Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # 28*28*1-->28*28*32
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)  # 14*14*32-->14*14*64
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Data
train_data = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=Batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=Batch_size, shuffle=True)

# Model definition
use_cuda = True
device = torch.device('cuda' if use_cuda else 'cpu')
Model = Net().to(device)
# optimizer & loss
optimizer = optim.Adam(Model.parameters(), lr=Learning_rate)
loss_func = nn.CrossEntropyLoss()


# Train
def train(Model, device):
    Model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Clean gradient
        optimizer.zero_grad()
        output = Model(data)
        loss = loss_func(output, target)
        # backward
        loss.backward()
        # update parameter in Model
        optimizer.step()

        # print information
        if batch_idx & 100 == 0:
            print(
                'Train Epoch: {},\t[{}/{} {:.0f}]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()
                )
            )


# Test
def test(Model, device):
    Model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = Model(data)
            test_loss += loss_func(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            accuracy += prediction.eq(target.view_as(prediction)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            'Test Average Loss: {:.6f}\tAccuracy: {}/{} {:.0f}%'.format(
                test_loss, accuracy, len(test_loader.dataset), 100. * accuracy / len(test_loader.dataset)
            )
        )
    pass


for epoch in range(1, Epoch + 1):
    train(Model, device)
    test(Model, device)
