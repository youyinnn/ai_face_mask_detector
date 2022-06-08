from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
import ssl
import helper
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context


num_epochs = 4
num_classes = 5
learning_rate = 0.001

train_dir ='./aug_1/'
#mutli-steps
transform = transforms.Compose(
       [transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                  download=True, transform=transform)
trainset = torchvision.datasets.ImageFolder(train_dir,transform=transform)
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,
#                                    shuffle=True, num_workers=2)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=32,
                                shuffle=True, num_workers=4)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
 #                                download=True, transform=transform)

#test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
 #                                  shuffle=False, num_workers=2)

classes = ('cloth_mask', 'mask_worn_incorrectly', 'n95_mask', 'no_face_mask',
            'surgical_mask')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(64 * 64 * 64, 1000), nn.ReLU(inplace=True),
                                  nn.Linear(1000, 512), nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Linear(512, 10)
                                  )

    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(-1, (64 * 64 * 64))

        # fc layer
        x = self.fc_layer(x)

        return x

model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # Forward pass
        #print(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backprop and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / total)


        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), (correct / total) * 100))

model.eval()
"""
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %' .format((correct / total) * 100))
"""
torch.save(model.state_dict(), './')