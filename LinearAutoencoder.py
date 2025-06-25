#Import
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
batch_size = 128
lr= 0.9
num_epochs = 5
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


# Training loop for different learning rate
     
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
print(f"Train MSE: {avg_loss:.4f}")
print(f"Test MSE: {avg_test_loss:.4f}")
train_array = np.array(training_loss)
test_array = np.array(tests_loss)

np.savez(f'training_curves/losses_lr{lr:.3e}.npz',  train=train_array,  test=test_array)


#Plotting the best learning rate
epochs = range(1, num_epochs+1)
print("Epochs length:", len(epochs))
plt.plot(epochs, training_loss, label="Training Loss")
plt.plot(epochs, tests_loss, label="Testing Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training vs Testing Loss for Best LR = lr")
plt.legend()
plt.grid(True)
plt.show()

fig =plt.figure()
plt.savefig