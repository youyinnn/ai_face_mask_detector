import copy

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as functions
import sklearn
import sklearn.model_selection
import Models
from torch.utils.data import DataLoader, TensorDataset

from data_process.DatasetHelper import label_map
from data_process.DatasetHelper import ImageDataset
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    print("Using GPU!")
    device = 'cuda'
else:
    print("Using CPU :(")
    device = 'cpu'

# unzip the augmented dataset and load it
# data = ImageDataset('./dataset')
# #model = Models.LinearNet()
# data_loader = torch.utils.data.DataLoader(data, batch_size=10000, shuffle=True)
# for data,labels in data_loader:
#     data_X,data_y = data,labels
#
# data_y = functions.one_hot(data_y)
# print(data_X.shape, data_y.shape)
def train_net(model,splits = 5, num_epochs = 50, pretrained_model = None, lr = 1e-3):
    if pretrained_model:
        #TODO implement loading a model
        return

    # Split data into TEST and TRAIN
    # Do not touch test until eval time
    X,y = get_all_data()
    X_train_val,X_test,y_train_val,y_test = sklearn.model_selection.train_test_split(
        X,y,test_size = .2, random_state = 6721)
    print("Training data shape:", X_train_val.shape)
    print("Training targets shape:", y_train_val.shape)

    accs = kfold_cross_validation(Models.LinearNet(), X, X_train_val, num_epochs, splits, y, y_train_val, lr)
    print(accs)

# Performs a kfold cross validation on a copy of the passed architecture
def kfold_cross_validation(base_model, X, X_train_val, num_epochs, splits, y, y_train_val, lr):
    # Now use training data to perform 5-fold cross validation
    # This will be used to tune hyper parameters and architecture
    # Stratified K-fold ensures balanced class representation
    SKF = sklearn.model_selection.StratifiedKFold(n_splits=splits)
    accuracies = [0] * splits
    i = 0
    # Perform the k-fold cross validation

    for train_index, val_index in SKF.split(X_train_val, torch.argmax(y_train_val, dim=1)):
        model_to_train = copy.deepcopy(base_model).to(device)
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        kfold_dataset = TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(kfold_dataset, batch_size=256, shuffle=True)
        trained_model = train_model(model_to_train, data_loader, num_epochs=num_epochs,lr=lr)
        acc = test_model(trained_model, X_val, y_val)
        accuracies[i] = acc
        print('Fold accuracy: ', acc)
        # print(accuracies[i])
        i = i + 1
    return accuracies


def test_model(model, X, y):
    X = X.to(device)
    y = y.to(device)
    #print("y_val shape:", y.shape)
    y_pred = model(X).to(device)
    #print("First 5 predictions and actual values, then accuracy")
    #print(y_pred[0:5], y[0:5])
    #print (sklearn.metrics.accuracy_score(y[0:5].argmax(dim=1).cpu().detach().numpy(), y_pred[0:5].argmax(dim=1).cpu().detach().numpy()) )
    accuracy = sklearn.metrics.accuracy_score(y.argmax(dim=1).cpu().detach().numpy(), y_pred.argmax(dim=1).cpu().detach().numpy())
    return accuracy

# retrieve the entire dataset and returns 2 tensors, 1 data and 1 targets
def get_all_data():
    data = ImageDataset('./dataset')
    # model = Models.LinearNet()
    # Using a batch size larger than the dataset means all data is retrieved in one loop iteration
    # Stretch goal: Make this work on arbitrarily large datasets by stacking the tensors in the data_loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=10000, shuffle=True)
    for data, labels in data_loader:
        data_X, data_y = data, labels

    data_y = functions.one_hot(data_y)
    print("All data shapes:", data_X.shape, data_y.shape)

    return data_X, data_y

def train_model(model, data_loader, num_epochs=50, lr=5e-4,):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print(f'Starting Epoch {epoch+1} of {num_epochs}')
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_pred = model(data).to(device)
            #print(target)
            #print(y_pred)
            loss = functions.binary_cross_entropy_with_logits(y_pred, target.float())
            loss.backward()
            optimizer.step()
    return model


#train_net(model,data_loader)
def hyper_param_tuning():

    train_net(Models.Base_CNN(), splits=5, num_epochs=50, lr=1e-3)

hyper_param_tuning()


