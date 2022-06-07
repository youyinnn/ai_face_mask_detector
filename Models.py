import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T



# A basic LinearNet to be used as a baseline and for debugging
class LinearNet(torch.nn.Module):
    def __init__(self, num_layers=3, layer_width=1000):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(3*256*256, layer_width), nn.ReLU(inplace=True)]
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

class Base_CNN(torch.nn.Module):
    def __init__(self, num_layers=2, layer_width=200):
        super(Base_CNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Conv2d(3, 48, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(48, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(3, stride=1)]
        self.layers += [nn.Conv2d(96, 96, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(96, 192, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(3, stride=1)]
        self.layers += [nn.Conv2d(192, 192, kernel_size=3), nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(192, 96, kernel_size=3, stride=2), nn.ReLU(inplace=True)]
        #self.linear1 = nn.LazyLinear(256)
        #self.linear2 = nn.Linear(512, 256)
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
        #x = self.linear1(x)
        x = self.fc(x)
        x = torch.softmax(x,dim=1)
        return x
