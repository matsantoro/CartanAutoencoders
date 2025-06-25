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


# Get all data in flat tensors
train_data, _ = next(iter(train_loader))
test_data, _ = next(iter(test_loader))

X_train = train_data.view(train_data.size(0), -1)  # (60000, 784)
X_test  = test_data.view(test_data.size(0), -1)    # (10000, 784)



#PCA
from sklearn.decomposition import PCA
import torch

n_components = 64  # match Linear AE hidden size
pca = PCA(n_components=n_components)
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()

# Fit on training data
X_train_pca = pca.fit_transform(X_train_np)
X_train_rec = pca.inverse_transform(X_train_pca)
X_train_rec = torch.tensor(X_train_rec).float()

# Transform and reconstruct test data
X_test_pca = pca.transform(X_test_np)
X_test_rec = pca.inverse_transform(X_test_pca)
X_test_rec = torch.tensor(X_test_rec).float()

# Compute MSE
train_mse_pca = torch.mean((X_train - X_train_rec)**2).item()
test_mse_pca = torch.mean((X_test - X_test_rec)**2).item()

print(f"PCA Train MSE: {train_mse_pca:.4f}")
print(f"PCA Test  MSE: {test_mse_pca:.4f}")


