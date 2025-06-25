#Impost MNIST
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch

from pathlib import Path

from itertools import chain

from tqdm import tqdm


#Set the data


dataset_train = MNIST(root=Path('data/'), download=True, train = True, transform=ToTensor())
train_dataloader = DataLoader(
    dataset_train,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)
dataset_test = MNIST(root=Path('data/'), download=True, train=False, transform=ToTensor())
test_dataloader = DataLoader(
    dataset_test,
    batch_size=10000,
)




#Euclidean Autoencoder
euclidean_encoder = torch.nn.Sequential(
    torch.nn.Linear(in_features=28*28, out_features=100),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=100, out_features=5),
    torch.nn.ReLU()
).cuda()

euclidean_decoder = torch.nn.Sequential(
    torch.nn.Linear(in_features=5, out_features=100),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=100, out_features=28*28),
    torch.nn.ReLU(),
).cuda()

#Hyberbolic Autoencoders
import sys

sys.path.append('CartanNetworks/code')

from models import HyperbolicNetwork
from layers import DmELU

hyp_encoder = HyperbolicNetwork(size=28*28, layer_size_list=[100+1, 2], activation=DmELU, head=torch.nn.Identity()).cuda()
hyp_decoder = HyperbolicNetwork(size=2, layer_size_list=[10 + 1, 28*28 + 1], activation=DmELU, head=torch.nn.Identity()).cuda()
#Training!
from geoopt.optim import RiemannianAdam

optimizer = torch.optim.Adam(params = chain(euclidean_encoder.parameters(), euclidean_decoder.parameters()),
                             lr = 1e-3, weight_decay=1e-4)

criterion = torch.nn.MSELoss()

epochs = 10

pbar = tqdm(range(epochs))

for epoch in pbar:
    train_losses = []
    for data, label in train_dataloader:
        optimizer.zero_grad()
        euclidean_encoder.zero_grad()
        euclidean_decoder.zero_grad()

        data = data.flatten(start_dim=1).cuda()
        latent = euclidean_encoder(data)
        recon = euclidean_decoder(latent)

        loss = criterion(data, recon)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.cpu().detach().numpy())
    
    tl = sum(train_losses)/len(train_losses)

    for data, _ in test_dataloader:
        with torch.no_grad():
            data = data.flatten(start_dim=1).cuda()
            recon = euclidean_decoder(euclidean_encoder(data))
            loss = criterion(data, recon)
    pbar.set_description(f"Train loss: {tl:.4f}, Test loss: {loss.item():.4f}")

import sys

sys.path.append('CartanNetworks/code')

from models import HyperbolicNetwork
from layers import DmELU

hyp_encoder = HyperbolicNetwork(size=28*28, layer_size_list=[100+1, 5], activation=DmELU, head=torch.nn.Identity()).cuda()
hyp_decoder = HyperbolicNetwork(size=5, layer_size_list=[10 + 1, 28*28 + 1], activation=DmELU, head=torch.nn.Identity()).cuda()


from geoopt.optim import RiemannianAdam

optimizer = RiemannianAdam(params = chain(hyp_encoder.parameters(), hyp_decoder.parameters()), lr=1e-3, weight_decay = 1e-4)


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

        loss = criterion(data, recon[..., 1:]) # only take the fiber

        loss.backward()
        optimizer.step()

        train_losses.append(loss.cpu().detach().numpy())
    
    tl = sum(train_losses)/len(train_losses)

    for data, _ in test_dataloader:
        with torch.no_grad():
            data = data.flatten(start_dim=1).cuda()
            recon = hyp_decoder(hyp_encoder(data))
            loss = criterion(data, recon[..., 1:])
    pbar.set_description(f"Train loss: {tl:.4f}, Test loss: {loss.item():.4f}")

#Plotting
import matplotlib.pyplot as plt

for item, _ in list(iter(test_dataloader)):
    for image in range(10):
        fig = plt.figure()
        axes = fig.subplots(nrows=1, ncols=3)
        axes[0].imshow(item[image,0,...].detach().cpu().numpy())
        axes[1].imshow(euclidean_decoder(euclidean_encoder(item.flatten(start_dim=1).cuda()))[image].reshape(28,28).cpu().detach().numpy())
        axes[2].imshow(hyp_decoder(hyp_encoder(item.flatten(start_dim=1).cuda()))[image, 1:].reshape(28,28).cpu().detach().numpy())
        if not image:
            axes[0].set_title('Original')
            axes[1].set_title('Rec eucl')
            axes[2].set_title('Rec hyp')
        fig.show()


# Save one photo with all original + reconstructions (Euclidean & Hyperbolic)
import matplotlib.pyplot as plt
import torch

# Get one batch of test data
item, _ = next(iter(test_dataloader))
data_flat = item.flatten(start_dim=1).cuda()

with torch.no_grad():
    eucl_recon = euclidean_decoder(euclidean_encoder(data_flat)).cpu()
    hyp_recon = hyp_decoder(hyp_encoder(data_flat)).cpu()

num_images = 10
fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(8, 2.5 * num_images))

for i in range(num_images):
    # Original
    axes[i, 0].imshow(item[i, 0].numpy(), cmap='gray')
    axes[i, 0].axis('off')
    if i == 0:
        axes[i, 0].set_title('Original')

    # Euclidean reconstruction
    axes[i, 1].imshow(eucl_recon[i].reshape(28, 28).numpy(), cmap='gray')
    axes[i, 1].axis('off')
    if i == 0:
        axes[i, 1].set_title('Rec Eucl')

    # Hyperbolic reconstruction
    axes[i, 2].imshow(hyp_recon[i, 1:].reshape(28, 28).numpy(), cmap='gray')
    axes[i, 2].axis('off')
    if i == 0:
        axes[i, 2].set_title('Rec Hyp')

plt.tight_layout()
plt.savefig("comparison_results.png")  # one photo saved here
plt.show()
