import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T



# A basic LinearNet to be used as a baseline and for debugging
class LinearNet(torch.nn.Module):
    def __init__(self, num_layers=2, layer_width=100):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.LazyLinear(layer_width), nn.ReLU(inplace=True)]
        for i in range(num_layers):
            self.layers += [nn.Linear(layer_width, layer_width), nn.ReLU(inplace=True)]
        #self.layers += [nn.LazyLinear(64), nn.ReLU(inplace=True)]
        self.task_head = nn.Linear(layer_width, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x starts with shape (N,C,H,W)
        x = x.reshape(x.shape[0], -1).float()

        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = self.task_head(x)
        #x = self.softmax(x)
        return x

class Big_CNN(torch.nn.Module):
    def __init__(self):
        super(Big_CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Conv2d(3, 48, kernel_size=7), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(48, 72, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(72, 72, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        #self.layers += [nn.Conv2d(72, 48, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 192, kernel_size=3), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 256)
        # self.layers += [nn.LazyLinear(32), nn.ReLU(inplace=True)]
        self.fc = nn.LazyLinear(5)

    def forward(self, x):
        N,C,H,W = x.shape
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = self.layers[0](x)
        #x = x.reshape(x.shape[0], x.shape[1], 56, 56)
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = x.view(N, -1)
        x = self.linear1(x)
        x = self.fc(x)
        # x = torch.softmax(x,dim=1)
        return x

# A recreation of the famous AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 227 * 227 * 3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # 55 * 55 * 96
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 27 * 27 * 96
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 27 * 27 * 256
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 13 * 13 * 256
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 13 * 13 * 384
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
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

# Accuracy ~66%
class Base_CNN(torch.nn.Module):
    def __init__(self):
        super(Base_CNN, self).__init__()

        self.name = "Base_CNN"
        self.layers = nn.ModuleList()
        self.layers += [nn.Conv2d(3, 6, kernel_size=3), nn.ReLU(inplace=True)]
        # self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(6, 12, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(12, 24, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(24, 48, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(48, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]

        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(192, 192, kernel_size=3), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 256)
        # self.layers += [nn.LazyLinear(32), nn.ReLU(inplace=True)]
        self.fc = nn.LazyLinear(5)

    def forward(self, x):
        N,C,H,W = x.shape
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = self.layers[0](x)
        #x = x.reshape(x.shape[0], x.shape[1], 56, 56)
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = x.view(N, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)
        # x = torch.softmax(x,dim=1)
        return x

class Less_Pooling_CNN(torch.nn.Module):
    def __init__(self):
        self.name = "Less_Pooling"
        super(Less_Pooling_CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Conv2d(3, 6, kernel_size=3), nn.ReLU(inplace=True)]
        # self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(6, 12, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(12, 24, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(24, 48, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(48, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]

        #self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(192, 192, kernel_size=3), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 256)
        # self.layers += [nn.LazyLinear(32), nn.ReLU(inplace=True)]
        self.fc = nn.LazyLinear(5)

    def forward(self, x):
        N,C,H,W = x.shape
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = self.layers[0](x)
        #x = x.reshape(x.shape[0], x.shape[1], 56, 56)
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = x.view(N, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)
        # x = torch.softmax(x,dim=1)
        return x

class Less_Conv_CNN(torch.nn.Module):
    def __init__(self):
        super(Less_Conv_CNN, self).__init__()

        self.name = "Less_Conv"
        self.layers = nn.ModuleList()
        self.layers += [nn.Conv2d(3, 6, kernel_size=3), nn.ReLU(inplace=True)]
        #self.layers += [nn.MaxPool2d(2, stride=1)]
        #self.layers += [nn.Conv2d(6, 12, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        #self.layers += [nn.Conv2d(12, 24, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(6, 48, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(48, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]

        self.layers += [nn.MaxPool2d(2, stride=1)]
        self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 192, kernel_size=3), nn.ReLU(inplace=True)]
        #self.layers += [nn.Conv2d(192, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.linear1 = nn.LazyLinear(256)
        self.linear2 = nn.Linear(256, 256)
        # self.layers += [nn.LazyLinear(32), nn.ReLU(inplace=True)]
        self.fc = nn.LazyLinear(5)

    def forward(self, x):
        N,C,H,W = x.shape
        #x = x.reshape(x.shape[0], x.shape[1], -1)
        # x = self.layers[0](x)
        #x = x.reshape(x.shape[0], x.shape[1], 56, 56)
        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = x.view(N, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.fc(x)
        # x = torch.softmax(x,dim=1)
        return x