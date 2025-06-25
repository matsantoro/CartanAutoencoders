# Imports
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from itertools import product
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from itertools import chain
from tqdm import tqdm
sys.path.append('CartanNetworks/code')
from models import HyperbolicNetwork
from layers import DmELU
from geoopt.optim import RiemannianAdam

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
dataset_train = MNIST(root=Path('data/'), download=True, train=True, transform=ToTensor())
train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
dataset_test = MNIST(root=Path('data/'), download=True, train=False, transform=ToTensor())
test_dataloader = DataLoader(dataset_test, batch_size=10000)

# Euclidean autoencoder builder
def build_euclidean_autoencoder(hidden_layers, latent_dim=5):
    layers_encoder = []
    input_dim = 28*28
    for h in hidden_layers:
        layers_encoder.append(torch.nn.Linear(input_dim, h))
        layers_encoder.append(torch.nn.ReLU())
        input_dim = h
    layers_encoder.append(torch.nn.Linear(input_dim, latent_dim))
    layers_encoder.append(torch.nn.ReLU())
    encoder = torch.nn.Sequential(*layers_encoder).to(device)

    layers_decoder = []
    input_dim = latent_dim
    for h in reversed(hidden_layers):
        layers_decoder.append(torch.nn.Linear(input_dim, h))
        layers_decoder.append(torch.nn.ReLU())
        input_dim = h
    layers_decoder.append(torch.nn.Linear(input_dim, 28*28))
    layers_decoder.append(torch.nn.ReLU())
    decoder = torch.nn.Sequential(*layers_decoder).to(device)

    return encoder, decoder

# Hyperbolic autoencoder builder
def build_hyperbolic_autoencoder(hidden_layers, latent_dim=5):
    layer_list_enc = [h + 1 for h in hidden_layers] + [latent_dim]
    encoder = HyperbolicNetwork(size=28*28, layer_size_list=layer_list_enc, activation=DmELU, head=torch.nn.Identity()).to(device)
    layer_list_dec = [10 + 1] + [28*28 + 1]
    decoder = HyperbolicNetwork(size=latent_dim, layer_size_list=layer_list_dec, activation=DmELU, head=torch.nn.Identity()).to(device)
    return encoder, decoder


# Training Euclidean
def train_euclidean(encoder, decoder, train_loader, lr, epochs=10):
    optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    encoder.train(); decoder.train()
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        for data, _ in train_loader:
            data = data.flatten(start_dim=1).to(device)
            optimizer.zero_grad()
            latent = encoder(data)
            recon = decoder(latent)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(sum(losses)/len(losses))
    return epoch_losses


# Training Hyperbolic
def train_hyperbolic(encoder, decoder, train_loader, lr, epochs=10):
    optimizer = RiemannianAdam(chain(encoder.parameters(), decoder.parameters()), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    encoder.train(); decoder.train()
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        for data, _ in train_loader:
            data = data.flatten(start_dim=1).to(device)
            optimizer.zero_grad()
            latent = encoder(data)
            recon = decoder(latent)
            loss = criterion(recon[..., 1:], data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(sum(losses)/len(losses))
    return epoch_losses


# Evaluation function 
def evaluate_loss(encoder, decoder, test_loader, is_hyperbolic=False):
    encoder.eval(); decoder.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.flatten(start_dim=1).to(device)
            recon = decoder(encoder(data))
            if is_hyperbolic:
                recon = recon[..., 1:]
            loss = criterion(recon, data)
            return loss.item(), recon.cpu(), data.cpu()


# Visualization
def visualize_reconstructions(orig_batch, eucl_batch, hyp_batch, lr, layers_str, eucl_loss, hyp_loss):
    fig, axes = plt.subplots(len(orig_batch), 3, figsize=(6, len(orig_batch)*2))
    for i in range(len(orig_batch)):
        axes[i, 0].imshow(orig_batch[i].numpy().reshape(28, 28), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(eucl_batch[i].numpy().reshape(28, 28), cmap='gray')
        axes[i, 1].set_title("Euclidean")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(hyp_batch[i].numpy().reshape(28, 28), cmap='gray')
        axes[i, 2].set_title("Hyperbolic")
        axes[i, 2].axis('off')

    fig.suptitle(f"LR={lr} Layers={layers_str}\nReconstruction Loss - Eucl: {eucl_loss:.4f}, Hyp: {hyp_loss:.4f}", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"reconstructions_lr{lr}_layers{layers_str.replace(' ', '_')}.png")
    plt.close()


# Hyperparameters
learning_rates = [1e-1]#,o.9,1]
layer_configs = [[100], [100, 50]]#,[100,50,25]]
epochs = 3

# Store results
all_train_losses = {}
all_test_losses = {}

# Loop
for lr, layers in product(learning_rates, layer_configs):
    print(f"\n=== Experiment: LR={lr}, Layers={layers} ===")
    eucl_encoder, eucl_decoder = build_euclidean_autoencoder(layers)
    hyp_encoder, hyp_decoder = build_hyperbolic_autoencoder(layers)

    eucl_train, hyp_train = [], []
    eucl_test, hyp_test = [], []

    for epoch in range(epochs):
        e_loss = train_euclidean(eucl_encoder, eucl_decoder, train_dataloader, lr, epochs=1)[0]
        h_loss = train_hyperbolic(hyp_encoder, hyp_decoder, train_dataloader, lr, epochs=1)[0]
        eucl_train.append(e_loss)
        hyp_train.append(h_loss)

        e_test_loss, _, _ = evaluate_loss(eucl_encoder, eucl_decoder, test_dataloader)
        h_test_loss, _, _ = evaluate_loss(hyp_encoder, hyp_decoder, test_dataloader, is_hyperbolic=True)
        eucl_test.append(e_test_loss)
        hyp_test.append(h_test_loss)

        print(f"Epoch {epoch+1}/{epochs}: Euclidean Train={e_loss:.4f} Test={e_test_loss:.4f} | Hyperbolic Train={h_loss:.4f} Test={h_test_loss:.4f}")

    # Save
    key_e = f"Euclidean_lr{lr}_layers{'-'.join(map(str,layers))}"
    key_h = f"Hyperbolic_lr{lr}_layers{'-'.join(map(str,layers))}"
    all_train_losses[key_e] = eucl_train
    all_test_losses[key_e] = eucl_test
    all_train_losses[key_h] = hyp_train
    all_test_losses[key_h] = hyp_test

    # Visualize reconstructions
    _, e_recon, orig = evaluate_loss(eucl_encoder, eucl_decoder, test_dataloader)
    _, h_recon, _ = evaluate_loss(hyp_encoder, hyp_decoder, test_dataloader, is_hyperbolic=True)
    visualize_reconstructions(orig[:10], e_recon[:10], h_recon[:10], lr, f"{layers}", eucl_test[-1], hyp_test[-1])
    import csv

    # Create the file
    filename = f"losses_lr{lr}_layers{'-'.join(map(str, layers))}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Euclidean Train Loss", "Hyperbolic Train Loss", "Euclidean Test Loss", "Hyperbolic Test Loss"])
        for et, ht, eTe, hTe in zip(eucl_train, hyp_train, eucl_test, hyp_test):
           writer.writerow([et, ht, eTe, hTe])
    # Save the losses
    visualize_reconstructions(orig[:10], e_recon[:10], h_recon[:10], lr, f"{layers}", eucl_test[-1], hyp_test[-1])


# Plot train loss
plt.figure(figsize=(12, 6))
for key, loss in all_train_losses.items():
    plt.plot(range(1, epochs+1), loss, label=key)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_curves.png")
plt.show()

# Plot test loss
plt.figure(figsize=(12, 6))
for key, loss in all_test_losses.items():
    plt.plot(range(1, epochs+1), loss, label=key)
plt.title("Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig("test_loss_curves.png")
plt.show()
