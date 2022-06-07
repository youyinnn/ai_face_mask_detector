import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T



#A basic LinearNet to be used as a baseline and for debugging
class LinearNet(torch.nn.Module):
    def __init__(self, num_layers=2, layer_width=200):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.Linear(3*256*256,layer_width), nn.ReLU(inplace=True)]
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