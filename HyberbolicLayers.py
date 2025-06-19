#Impost MNIST
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch

from pathlib import Path

from itertools import chain

from tqdm import tqdm
#Set the data
dataset_train = MNIST(roo
t=Path('data/'), download=True, train = True, transform=ToTensor())
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
#Hyberbolic Autoencoders
import sys

sys.path.append('CartanNetworks/code')

from models import HyperbolicNetwork
from layers import DmELU

hyp_encoder = HyperbolicNetwork(size=28*28, layer_size_list=[100+1, 2], activation=DmELU, head=torch.nn.Identity()).cuda()
hyp_decoder = HyperbolicNetwork(size=2, layer_size_list=[10 + 1, 28*28 + 1], activation=DmELU, head=torch.nn.Identity()).cuda()
#Training!
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