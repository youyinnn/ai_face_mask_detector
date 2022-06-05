import torch
import torch.nn as nn

#A basic LinearNet to be used as a baseline and for debugging
class LinearNet(torch.nn.Module):
    def __init__(self, num_layers=2, layer_width=1000):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers += [nn.LazyLinear(layer_width), nn.ReLU(inplace=True)]
        for i in range(self.hyper_params['num_layers']):
            self.layers += [nn.Linear(self.hyper_params['layer_width'], self.hyper_params['layer_width']), nn.ReLU(inplace=True)]
        #self.layers += [nn.LazyLinear(64), nn.ReLU(inplace=True)]
        self.task_head = nn.Linear(layer_width, 5)

    def forward(self, x):
        # x starts with shape (N,C,H,W) or (10,1,28,28)
        N = x.shape[0]
        x = x.reshape(x.shape[0], -1)

        for i in range(0, len(self.layers)):
            x = self.layers[i](x)
        x = self.task_head(x)
        return x