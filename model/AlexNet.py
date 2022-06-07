import torch
from torch import nn
import os
import torch.nn.functional as F
import numpy as np


class AlexNet1(nn.Module):
    def __init__(self):
        super(AlexNet1, self).__init__()
        # 227 * 227 * 3
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # 55 * 55 * 96
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 27 * 27 * 96
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 27 * 27 * 256
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 13 * 13 * 256
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 256
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 6 * 6 * 256

        # fc
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet2(nn.Module):
    def __init__(self):
        super(AlexNet2, self).__init__()
        # 227 * 227 * 3
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # 55 * 55 * 96
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 27 * 27 * 96
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 27 * 27 * 256
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 13 * 13 * 256
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 256
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        # add maxpool here
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 6 * 6 * 384
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 6 * 6 * 256
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)
        # 5 * 5 * 256

        # fc
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=6400, out_features=3200)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=3200, out_features=3200)
        self.fc3 = nn.Linear(in_features=3200, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool3(x)
        x = F.relu(self.conv5(x))
        x = self.maxpool4(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet3(nn.Module):
    def __init__(self):
        super(AlexNet3, self).__init__()
        # 227 * 227 * 3
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # 55 * 55 * 96
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 27 * 27 * 96
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 27 * 27 * 256
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 13 * 13 * 256
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        # add conv layer here
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv6 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 256
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 6 * 6 * 256

        # fc
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CompositeAlexNet(nn.Module):

    def __init__(self, variant=None, loadkws={}):

        super(CompositeAlexNet, self).__init__()

        self.composite = load(variant, loadkws)

    def eval(self):
        super().eval()

        for folds in self.composite:
            folds.eval()

    def forward(self, x):

        rs = []

        for i, fold in enumerate(self.composite):
            # print(i, fold)
            rs.append(fold(x))

        rs = (rs[0] + rs[1] + rs[2] + rs[3] + rs[4]) / 5

        return rs

    def cuda(self):
        super().cuda()

        for i, fold in enumerate(self.composite):
            self.composite[i] = fold.cuda()

    # def predict


def load(variant=None, loadkws={}):
    net_name = 'net1' if variant == None else (
        'net2' if variant == 'net2' else 'net3')

    net_class = AlexNet1 if variant == None else (
        AlexNet2 if variant == 'net2' else AlexNet3)

    model_f1 = net_class()
    model_f1.load_state_dict(torch.load(os.path.join(
        os.getcwd(), 'model/alex_net', net_name, f'alex_{net_name}_f1.pth'), **loadkws))
    model_f2 = net_class()
    model_f2.load_state_dict(torch.load(os.path.join(
        os.getcwd(), 'model/alex_net', net_name, f'alex_{net_name}_f2.pth'), **loadkws))
    model_f3 = net_class()
    model_f3.load_state_dict(torch.load(os.path.join(
        os.getcwd(), 'model/alex_net', net_name, f'alex_{net_name}_f3.pth'), **loadkws))
    model_f4 = net_class()
    model_f4.load_state_dict(torch.load(os.path.join(
        os.getcwd(), 'model/alex_net', net_name, f'alex_{net_name}_f4.pth'), **loadkws))
    model_f5 = net_class()
    model_f5.load_state_dict(torch.load(os.path.join(
        os.getcwd(), 'model/alex_net', net_name, f'alex_{net_name}_f5.pth'), **loadkws))

    return model_f1, model_f2, model_f3, model_f4, model_f5
