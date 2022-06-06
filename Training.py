import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as functions
#Can change this
import Models

device = 'coda'
from data_process.DatasetHelper import label_map
from data_process.DatasetHelper import ImageDataset
import matplotlib.pyplot as plt


# unzip the augmented dataset and load it
data = ImageDataset('./dataset')
#model = Models.LinearNet()
data_loader = torch.utils.data.DataLoader(data, batch_size=10000, shuffle=True)
for data,labels in data_loader:
    data_X,data_y = data,labels

data_y = functions.one_hot(data_y)
print(data_X.shape, data_y.shape)
#Need to turn the dataset into one stack so I can split it
def train_net(model, data_loader, num_epochs=50, lr=1e-3,):
    dataset,labels = data_loader

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # for epoch in range(num_epochs):
    #     if epoch % 5 == 0:
    #         print(f'Starting Epoch {epoch+1} of {num_epochs}')
    #     for data, target in data_loader:
    #         data = data.to(device)
    #         target = target.to(device)
    #         optimizer.zero_grad()
    #
    #         y_pred = model(data)
    #         loss_function = nn.CrossEntropyLoss()
    #         loss = loss_function(target,y_pred)
    #         loss.backward()
    #         optimizer.step()


#train_net(model,data_loader)




