import copy
import os

import numpy as np
import torchvision.transforms as T

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as functions
import sklearn
import sklearn.model_selection
import evaluation
import Models
from torch.utils.data import DataLoader, TensorDataset

from data_process.DatasetHelper import label_map
from data_process.DatasetHelper import ImageDataset
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    #print("Using GPU!")
    device = 'cuda'
    load_kws = {}
else:
    #print("Using CPU :(")
    load_kws = {
        'map_location': torch.device('cpu')
    }
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
def train_net(model_type, tuning = False, load_path = None, splits = 5, num_epochs = 50, lr = 1e-3, batch_size = 128):

    # Split data into TEST and TRAIN
    # Do not touch test until eval time
    X,y = get_all_data()
    #Setting random state means the training/validation and test images are the same between runs
    X_train_val,X_test,y_train_val,y_test = sklearn.model_selection.train_test_split(
        X,y,test_size=.2, stratify=y, random_state=6721)
    print("Training data shape:", X_train_val.shape)
    print("Training targets shape:", y_train_val.shape)
    if tuning:
    #Perform K-fold cross validation. This is used for tuning our model
        accs = kfold_cross_validation(model_type, X_train_val, num_epochs, splits, y_train_val, lr, batch_size = batch_size)
        return accs
    else:
    # After tuning, train on entire train/validation set before testing on the test set
        final_train_dataset = TensorDataset(X_train_val, y_train_val)
        data_loader = torch.utils.data.DataLoader(final_train_dataset, batch_size=128, shuffle=True)
        if load_path:
            trained_model = load_model(model_type, load_path).to(device)
        else:
            trained_model = train_model(model_type().to(device), data_loader,
                                        x_val=X_test, y_val=y_test,
                                        num_epochs=num_epochs,
                                        track_training=True)
        print("Test accuracy: ", test_model(trained_model, X_test, y_test))
        metrics = eval_model(trained_model, X_test, y_test)
        print("Final Report:", metrics['report'])
        np.save('Final_Test_Metrics' + trained_model.name, metrics)
        return trained_model


# Performs a kfold cross validation on a copy of the passed architecture
def kfold_cross_validation(base_model, X_train_val, num_epochs, splits, y_train_val, lr, batch_size=128):
    # Now use training data to perform 5-fold cross validation
    # This will be used to tune hyper parameters and architecture
    # Stratified K-fold ensures balanced class representation
    SKF = sklearn.model_selection.StratifiedKFold(n_splits=splits)
    accuracies = [0] * splits
    i = 0
    # Perform the k-fold cross validation

    for train_index, val_index in SKF.split(X_train_val, torch.argmax(y_train_val, dim=1)):
        model_to_train = base_model().to(device)
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]
        kfold_dataset = TensorDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(kfold_dataset, batch_size=batch_size, shuffle=True)
        trained_model = train_model(model_to_train, data_loader,
                                    x_val=X_val, y_val=y_val,
                                    num_epochs=num_epochs, lr=lr)
        print("Evaluation metrics for this fold:")
        print(eval_model(trained_model,X_val,y_val)['report'])
        acc = test_model(trained_model, X_val, y_val)
        accuracies[i] = acc
        # print('Fold accuracy: ', acc)
        # print(accuracies[i])
        i = i + 1
    return accuracies


def test_model(model, X, y):
    X = X.to(device)
    y = y.to(device)
    test_data = TensorDataset(X, y)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    #print("y_val shape:", y.shape)
    overall_accuracy = 0
    total_samples = 0
    for data, labels in test_loader:
        y_pred = model(data)
        num_samples = data.shape[0]
        accuracy = sklearn.metrics.accuracy_score(labels.argmax(dim=1).cpu().detach().numpy(), y_pred.argmax(dim=1).cpu().detach().numpy())
        overall_accuracy = overall_accuracy + accuracy * num_samples
        total_samples = total_samples + num_samples

    return overall_accuracy/total_samples

# retrieve the entire dataset and returns 2 tensors, 1 data and 1 targets
def get_all_data(resize = 128):
    # Size set to 128 to prevent memory issues
    transform_train = T.Compose([T.Resize((resize, resize))])
    data = ImageDataset(os.path.join(os.getcwd(), 'data', 'aug_1'), transform=transform_train)
    # model = Models.LinearNet()
    # Using a batch size larger than the dataset means all data is retrieved in one loop iteration
    # Stretch goal: Make this work on arbitrarily large datasets by stacking the tensors in the data_loader
    data_loader = torch.utils.data.DataLoader(data, batch_size=10000, shuffle=False)
    for data, labels in data_loader:
        data_X, data_y = data.float(), labels

    data_y = functions.one_hot(data_y)
    print("All data shapes:", data_X.shape, data_y.shape)

    return data_X, data_y

def train_model(model, data_loader,x_val=None,y_val=None, num_epochs=50, lr=1e-4, track_training = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_metrics = np.zeros((num_epochs,4))
    model.train()
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
        #last_acc = test_model(model,x_val,y_val)
        #print(f"Epoch {epoch} accuracy: {last_acc*100}%")
        if track_training:
            metrics = eval_model(model,x_val,y_val)
            all_metrics[epoch] = np.array((metrics['acc'], metrics['pre'], metrics['rec'],metrics['f1']))
    if track_training:
        np.save('Metrics_Tracking_'+model.name, all_metrics)
    return model

# Performs the hyperparameter tuning using random search.
# Learning Rate and Num_Epochs used for an example
def hyper_parameter_tuning(model_type, n_trials):
    lrs = torch.randint(low=1, high=20, size=(n_trials,)) * (1e-4)
    num_epochs = torch.randint(low=25, high=50, size=(n_trials,))
    mean_accs = [0]*n_trials
    for i,(lr,epochs) in enumerate(zip(lrs,num_epochs)):
        print(f'lr = {lr}, epochs = {epochs}')
        accs = train_net(model_type, tuning=True, lr=lr, num_epochs=epochs)
        mean_accs[i] = sum(accs)/5
    print(lrs)
    print(num_epochs)
    print(mean_accs)

# Helper method to get predictions from model output
def output_to_label(output):
    print(output)
    rs = output.argmax(axis=1)
  # [1,2,3,100,1] -> 3
    return rs

# Get evaluation metrics on a trained model given a test set
def eval_model(model, x,y):
    test_data = TensorDataset(x, y)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)
    return evaluation.evaluate(test_loader, model, output_to_label)
#hyper_parameter_tuning(Models.Base_CNN,3)
#accs = train_net(Models.Base_CNN, tuning=True, splits=5, num_epochs=50, lr=1e-4, batch_size=128)
#print("Accuracies: ", accs, sum(accs)/5)

# load a model from a file
def load_model(model_type, PATH):
    model = model_type()
    print(load_kws)
    model.load_state_dict(torch.load(PATH, **load_kws))
    return model

# Perform final training after tuning and save the model
def train_final_model(model_type, PATH, num_epochs=50, lr=1e-4, batch_size=128):
    model = train_net(model_type, tuning=False, num_epochs=num_epochs, lr=lr, batch_size=batch_size)
    PATH = PATH + '_' + model.name
    print("Saving model!")
    torch.save(model.state_dict(), PATH)

# Load a model and test its accuracy to amke sure it loaded correctly
def load_and_run_model(model_type,PATH):
    print("Loading model!")
    train_net(model_type, tuning=False, load_path=PATH)


#train_final_model(Models.Base_CNN, 'Final_Model',)

#load_and_run_model(Models.Base_CNN, 'Final_Model_Base_CNN')

#train_final_model(Models.Less_Conv_CNN, 'Final_Model',)

#load_and_run_model(Models.Less_Conv_CNN, 'Final_Model_Less_Conv')

#train_final_model(Models.Less_Pooling_CNN, 'Final_Model',)

#load_and_run_model(Models.Less_Pooling_CNN, 'Final_Model_Less_Pooling')