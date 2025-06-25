import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
batch_size = 128
learning_rates = [0.9,0.1,0.01]
num_epochs = 10
hidden_dim = 64

# Dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Linear Autoencoder
class LinearAutoencoder(nn.Module):
    def __init__(self, hidden_dim):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Linear(28*28, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, 28*28)

    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Store results
test_mse_results = []
all_train_losses = {}
all_test_losses = {}

# Training loop for different learning rates
for lr in learning_rates:
     
    training_loss = []
    tests_loss = []
    print(f"\nTraining with learning rate: {lr}")
    model = LinearAutoencoder(hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images.view(-1, 28*28))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        training_loss.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}", end='\r')
        test_loss = 0
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model (images)
            loss = criterion(outputs, images.view(-1, 28*28))
            test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        tests_loss.append(avg_test_loss)
        all_train_losses[lr] = training_loss
        all_test_losses[lr] = tests_loss
        
         # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images.view(-1, 28*28))
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
    test_mse_results.append((lr, avg_test_loss))
    print(f"Test MSE for learning rate {lr}: {avg_test_loss:.4f}")
    train_array = np.array(training_loss)
    test_array = np.array(tests_loss)

    np.savez(f'training_curves/losses_lr{lr:.3e}.npz',  train=train_array,  test=test_array)


# Plotting
for lr, loss_array in all_train_losses.items():
    epochs = range(1, len(loss_array) + 1)
    plt.plot(epochs, loss_array, label=f"lr = {lr}")

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs for Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

#The best learning rate
best_lr = test_mse_results[np.argmin([x[1] for x in test_mse_results])][0]
print(f"Best learning rate: {best_lr}")



#Plotting the best learning rate
epochs = range(1, len(all_train_losses[best_lr]) + 1)
plt.plot(epochs, all_train_losses[best_lr], label="Training Loss")
plt.plot(epochs, all_test_losses[best_lr], label="Testing Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training vs Testing Loss for Best LR = {best_lr}")
plt.legend()
plt.grid(True)
plt.show()
fig =plt.figure()
plt.savefig