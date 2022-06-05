import torch
import torch.optim
import torch.nn.functional as functions
#Can change this
device = 'coda'
def train_net(model, x_train, y_train, num_epochs=50, lr=1e-3, batch_size=128):

    #Update this to use a real loader!
    train_loader = (x_train,y_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print(f'Starting Epoch {epoch+1} of {num_epochs}')
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            y_pred = model(data)
            loss = functions.cross_entropy(target,y_pred)
            loss.backward()
            optimizer.step()




