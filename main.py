import numpy as np
import pandas as pd
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import argparse


parser = argparse.ArgumentParser(description='')

parser.add_argument('--batchsize', type=int, default=128, help='batchsize of train')
parser.add_argument('--epochs', type=int, default=10, help='epochs of train')
parser.add_argument('--lr_rate', type=float, default=0.001, help='learing rate of train')
parser.add_argument('--use_gpu', type=bool, default=True, help='using gpu if True else using cpu for train')
parser.add_argument('--train_data', default='./Kannada-MNIST/train.csv', help='path of training data')
parser.add_argument('--val_data', default='./Kannada-MNIST/Dig-MNIST.csv', help='path of testing data')

args = parser.parse_args()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        data = np.array(pd.read_csv(datatxt))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        line = self.data[item]
        label = line[0]
        img = line[1:].reshape(28, 28, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 128, 4,4

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


data_tf = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5])])
data_tf2 = transforms.Compose(
                [transforms.ToTensor()])
train_data = MyDataset(args.train_data, transform=data_tf)
train_loader = DataLoader(dataset=train_data, batch_size=args.batchsize, shuffle=True)
val_data = MyDataset(args.val_data, transform=data_tf)
val_loader = DataLoader(dataset=val_data, batch_size=args.batchsize, shuffle=False)

device = torch.device("cuda" if args.use_gpu is True else "cpu")
net = CNN().to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr_rate)
criterion = nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    train_correct = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        output = net(data)
        loss = criterion(output, label)
        pred = output.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        if batch_idx % 20 == 0:
            print('Epoch:[{}/{}]  Step:[{}/{}]  Loss:{:.6f}'.format(
                epoch, args.epochs, (batch_idx+1)*len(data), len(train_loader.dataset),
                loss.item()
            ))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_correct = 0
    for data, label in val_loader:
        data, label = data.to(device), label.to(device)
        output = net(data)
        pred = output.data.max(1, keepdim=True)[1]
        test_correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    print('train_acc:{:.6f}\ttest_acc:{:.6f}\n'.format(
        train_correct.item() / len(train_loader.dataset),
        test_correct.item() / len(val_loader.dataset)
    ))
