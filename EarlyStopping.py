# Imports
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import torch
from pathlib import Path
from itertools import chain
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

# Set up data 
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from pathlib import Path

train_dataset = MNIST(root=Path('data/'), download=True, train=True, transform=ToTensor())
test_dataset = MNIST(root=Path('data/'), download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=10000)

# Import hyperbolic model
sys.path.append('CartanNetworks/code')
from models import HyperbolicNetwork
from layers import DmELU

hyp_encoder = HyperbolicNetwork(size=28*28, layer_size_list=[100+1, 5], activation=DmELU, head=torch.nn.Identity()).cuda()
hyp_decoder = HyperbolicNetwork(size=5, layer_size_list=[10 + 1, 28*28 + 1], activation=DmELU, head=torch.nn.Identity()).cuda()

# Optimizer and loss
from geoopt.optim import RiemannianAdam
criterion = torch.nn.MSELoss()
optimizer = RiemannianAdam(params=chain(hyp_encoder.parameters(), hyp_decoder.parameters()), lr=1e-3, weight_decay=1e-4)

# Training loop with early stopping
epochs = 100
early_stop_window = 10
test_losses = []
early_stopped = False
pbar = tqdm(range(epochs))

for epoch in pbar:
    train_losses = []
    for data, label in train_dataloader:
        optimizer.zero_grad()
        hyp_encoder.zero_grad()
        hyp_decoder.zero_grad()

        data = data.flatten(start_dim=1).cuda()
        latent = hyp_encoder(data)
        recon = hyp_decoder(latent)

        loss = criterion(data, recon[..., 1:])
        loss.backward()
        optimizer.step()

        train_losses.append(loss.cpu().detach().numpy())

    tl = sum(train_losses) / len(train_losses)

    # Test loss
    for data, _ in test_dataloader:
        with torch.no_grad():
            data = data.flatten(start_dim=1).cuda()
            recon = hyp_decoder(hyp_encoder(data))
            test_loss = criterion(data, recon[..., 1:])

    test_losses.append(test_loss.item())
    pbar.set_description(f"Train loss: {tl:.4f}, Test loss: {test_loss.item():.4f}")

    # Early stopping
    if len(test_losses) >= early_stop_window + 1:
        last_test = test_losses[-1]
        prev_window = test_losses[-early_stop_window-1:-1]
        if all(last_test >= prev for prev in prev_window):
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            early_stopped = True
            break

if not early_stopped:
    print(f"\n✅ Training completed all {epochs} epochs without early stopping.")

# Plot example reconstructions
for item, _ in list(iter(test_dataloader)):
    for image in range(10):
        fig = plt.figure()
        axes = fig.subplots(nrows=1, ncols=2)
        axes[0].imshow(item[image, 0, ...].detach().cpu().numpy(), cmap='gray')
        axes[1].imshow(hyp_decoder(hyp_encoder(item.flatten(start_dim=1).cuda()))[image, 1:].reshape(28, 28).cpu().detach().numpy(), cmap='gray')
        if not image:
            axes[0].set_title('Original')
            axes[1].set_title('Rec Hyp')
        fig.show()

# Save one photo with all original + hyperbolic reconstructions
item, _ = next(iter(test_dataloader))
data_flat = item.flatten(start_dim=1).cuda()

with torch.no_grad():
    hyp_recon = hyp_decoder(hyp_encoder(data_flat)).cpu()

num_images = 10
fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 2.5 * num_images))

for i in range(num_images):
    axes[i, 0].imshow(item[i, 0].numpy(), cmap='gray')
    axes[i, 0].axis('off')
    if i == 0:
        axes[i, 0].set_title('Original')

    axes[i, 1].imshow(hyp_recon[i, 1:].reshape(28, 28).numpy(), cmap='gray')
    axes[i, 1].axis('off')
    if i == 0:
        axes[i, 1].set_title('Rec Hyp')

plt.tight_layout()
plt.savefig("comparison_results.png")
plt.show()
