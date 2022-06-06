import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from data_process.DatasetHelper import ImageDataset
import torch.nn as nn

# Functional module contains helper functions
import torch.nn.functional as F
from torch.autograd import Variable

# unzip the augmented dataset and load it
training_data = ImageDataset('/Users/sonjack/Downloads/aug_1/')
print()
testing_data = ImageDataset('/Users/sonjack/Downloads/aug_1/', train=False)

batch_size = 40

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=1)


class Net(nn.Module):

    def __init__(self):
        super().__init__()

        # NOTE: All Conv2d layers have a default padding of 0 and stride of 1,
        # which is what we are using.

        # Convolution Layer 1                                           # 256 x 256 x 3   (input)
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5, padding=2)  # 256 x 256 x 20  (after 1st convolution)
        self.conv1_drop = nn.Dropout2d(p=0.5)  # Same as above
        self.maxpool1 = nn.MaxPool2d(2)  # 128 x 128 x 20  (after pooling)
        self.relu1 = nn.ReLU()  # Same as above

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5, padding=2)  # 128 x 128 x 30  (after 2nd convolution)
        self.conv2_drop = nn.Dropout2d(p=0.5)  # Same as above
        self.maxpool2 = nn.MaxPool2d(2)  # 64 x 64 x 30    (after pooling)
        self.relu2 = nn.ReLU()  # Same as above

        # Convolution Layer 3
        self.conv3 = nn.Conv2d(30, 20, kernel_size=5, padding=2)  # 64 x 64 x 30  (after 2nd convolution)
        self.conv3_drop = nn.Dropout2d(p=0.5)  # Same as above
        self.maxpool3 = nn.MaxPool2d(2)  # 32 x 32 x 30    (after pooling)
        self.relu3 = nn.ReLU()  # Same as above

        # Fully connected layers
        self.fc1 = nn.Linear(30720, 3840)
        self.fc2 = nn.Linear(3840, 480)
        self.fc3 = nn.Linear(480, 5)

    def forward(self, x):
        # Convolution Layer 1
        x = self.conv1(x)
        x = self.conv1_drop(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        # Convolution Layer 2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # Convolution Layer 3
        x = self.conv3(x)
        x = self.conv3_drop(x)
        x = self.maxpool3(x)
        x = self.relu3(x)

        # Switch from activation maps to vectors
        x = x.view(-1, 30720)

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)

        # Fully connected layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training=True)

        # Fully connected layer 3
        x = self.fc3(x)

        return x

# The model
net = Net()

# Our loss function
criterion = nn.CrossEntropyLoss()

# Our optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

num_epochs = 1

train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

for epoch in range(num_epochs):

    ############################
    # Train
    ############################

    iter_loss = 0.0
    correct = 0
    iterations = 0

    net.train()  # Put the network into training mode

    for i, (items, classes) in enumerate(train_loader):
        # Convert torch tensor to Variable
        items = Variable(items)
        classes = Variable(classes)

        optimizer.zero_grad()  # Clear off the gradients from any past operation
        outputs = net(items.float())  # Do the forward pass
        loss = criterion(outputs, classes)  # Calculate the loss
        iter_loss += loss.data  # Accumulate the loss
        loss.backward()  # Calculate the gradients with help of back propagation
        optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data).sum()
        iterations += 1

    # Record the training loss
    train_loss.append(iter_loss / iterations)
    # Record the training accuracy
    train_accuracy.append((100 * correct / len(train_loader.dataset)))

    ############################
    # Validate - How did we do on the unseen dataset?
    ############################

    loss = 0.0
    correct = 0
    iterations = 0

    net.eval()  # Put the network into evaluate mode

    for i, (items, classes) in enumerate(test_loader):
        # Convert torch tensor to Variable
        items = Variable(items)
        classes = Variable(classes)

        outputs = net(items)  # Do the forward pass
        loss += criterion(outputs, classes).data  # Calculate the loss

        # Record the correct predictions for training data
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == classes.data).sum()

        iterations += 1

    # Record the validation loss
    valid_loss.append(loss / iterations)
    # Record the validation accuracy
    valid_accuracy.append(correct / len(test_loader.dataset) * 100.0)

    print('Epoch %d/%d, Tr Loss: %.4f, Tr Acc: %.4f, Val Loss: %.4f, Val Acc: %.4f'
          % (epoch + 1, num_epochs, train_loss[-1], train_accuracy[-1],
             valid_loss[-1], valid_accuracy[-1]))