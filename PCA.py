#Import
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=len(train_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

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
