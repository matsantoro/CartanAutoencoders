# Imports
import torch
from torch.utils.data import DataLoader, Subset
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

results_folder = Path('results')

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST
dataset_train = MNIST(root=Path('data/'), download=True, train = True, transform=ToTensor())
train_dataloader = DataLoader( dataset_train,batch_size=64,shuffle=True,pin_memory=True,drop_last=True)
dataset_test = MNIST(root=Path('data/'), download=True, train=False, transform=ToTensor())
test_dataloader = DataLoader(dataset_test,batch_size=10000)
 
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
    layer_list_enc = hidden_layers + [latent_dim]
    encoder = HyperbolicNetwork(size=28*28, layer_size_list=layer_list_enc, activation=DmELU, head=torch.nn.Identity()).to(device)
    layer_list_dec = list(reversed(hidden_layers))[1:] + [28*28+1]
    decoder = HyperbolicNetwork(size=latent_dim, layer_size_list=layer_list_dec, activation=DmELU, head=torch.nn.Identity()).to(device)
    return encoder, decoder

# Training Euclidean
def train_euclidean(encoder, decoder, train_loader, optimizer, criterion):
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
    return sum(losses)/len(losses)

# Training Hyperbolic with Energy Tracking
def train_hyperbolic(encoder, decoder, train_loader, optimizer, criterion, track_energy=False):
    losses = []
    batch_squared_sums = []  # Store squared sums for all batches in this epoch
    
    for data, _ in train_loader:
        data = data.flatten(start_dim=1).to(device)
        optimizer.zero_grad()
        latent = encoder(data)
        recon = decoder(latent)
        loss = criterion(recon[..., 1:], data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        # Track energy if requested
        if track_energy:
            with torch.no_grad():
                encoder.eval()
                decoder.eval()
                
                # Get per-layer SQUARED SUMS for this batch
                batch_layer_sums = []
                
                # Track encoder layers
                repr_data = data
                from PGTS import HyperbolicAlgebra
                m = HyperbolicAlgebra()
                
                for layer in encoder.hidden:
                    repr_data = layer(repr_data)
                    # Sum of squared first coordinates for this batch
                    cartan_coords = m.cartan(repr_data)  # First coordinate
                    squared_sum = torch.sum(cartan_coords.pow(2)).item()
                    batch_layer_sums.append(squared_sum)
                
                # Track decoder layers  
                repr_latent = latent
                for layer in decoder.hidden:
                    repr_latent = layer(repr_latent)
                    # Sum of squared first coordinates for this batch
                    cartan_coords = m.cartan(repr_latent)  # First coordinate
                    squared_sum = torch.sum(cartan_coords.pow(2)).item()
                    batch_layer_sums.append(squared_sum)
                
                batch_squared_sums.append(batch_layer_sums)
                
                encoder.train()
                decoder.train()
    
    avg_loss = sum(losses)/len(losses)
    
    if track_energy and batch_squared_sums:
        # Average squared sums across all batches for this epoch
        epoch_avg_energies = []
        num_layers = len(batch_squared_sums[0])
        
        for layer_idx in range(num_layers):
            # Get squared sums from all batches for this layer
            layer_squared_sums = [batch[layer_idx] for batch in batch_squared_sums]
            # Average across batches
            avg_squared_sum = sum(layer_squared_sums) / len(layer_squared_sums)
            epoch_avg_energies.append(avg_squared_sum)
        
        return avg_loss, epoch_avg_energies
    
    return avg_loss


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
    plt.savefig(results_folder/f"reconstructions_lr{lr}_layers{layers_str.replace(' ', '_')}.png")
    plt.close()


# Hyperparameters
learning_rates = [1e-2,1e-3,1e-4]
layer_configs = [[100], [100, 50],[100,50,25]]
hidden_dimension_configs = [5,10,20]
epochs = 500
early_stop_window = 10
test_losses = []
euc_early_stopped = False
hyp_early_stopped = False
num_experiments=5

# Store results
all_train_losses = {}
all_test_losses = {}
all_layer_energies = {}  # New: Store energy tracking results


# Loops
for lr, layers, hd in product(learning_rates, layer_configs, hidden_dimension_configs):
    for exp in range(num_experiments):
        print(f"\n=== Experiment {exp+1}/{num_experiments}: LR={lr}, Layers={layers} ===")
        eucl_encoder, eucl_decoder = build_euclidean_autoencoder(layers, latent_dim=hd)
        hyp_encoder, hyp_decoder = build_hyperbolic_autoencoder(layers, latent_dim=hd)
        eucl_train, hyp_train = [], []
        eucl_test, hyp_test = [], []
        print("Euclidean Model")
        optimizer = torch.optim.Adam(chain(eucl_encoder.parameters(), eucl_decoder.parameters()), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        eucl_encoder.train(); eucl_decoder.train()
        for epoch in range(epochs):
            e_loss = train_euclidean(eucl_encoder, eucl_decoder, train_dataloader, optimizer, criterion)
            eucl_train.append(e_loss)
            e_test_loss, _, _ = evaluate_loss(eucl_encoder, eucl_decoder, test_dataloader)
            eucl_test.append(e_test_loss)
            print(f"Epoch {epoch+1}/{epochs}: Euclidean Train={e_loss:.4f} Test={e_test_loss:.4f}")
            # Early stopping
            if len(eucl_test) >= early_stop_window + 1 :
                last_eucl_test = eucl_test[-1]
                prev_eucl_window = eucl_test[-early_stop_window-1:-1]

                if all(last_eucl_test >= prev for prev in prev_eucl_window):
                    print(f"\n🛑 Early stopping for the Euclidean triggered at epoch {epoch + 1}")
                    euc_early_stopped = True
                    for i in range (epoch+1, 100):
                        eucl_test.append(last_eucl_test) 
                        eucl_train.append(eucl_train[-1])
                    break
        if not euc_early_stopped:
            print(f"\n✅ Training Euclidean Autoencoder completed all {epochs} epochs without early stopping.")



        #Hyperbolic loop
        print("Hyperbolic Model")
        optimizer = RiemannianAdam(chain(hyp_encoder.parameters(), hyp_decoder.parameters()), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()
        hyp_encoder.train(); hyp_decoder.train()
        
        # Initialize energy tracking for this experiment
        layer_energy_epochwise = []
        
        for epoch in range(epochs):
            h_loss, epoch_energies = train_hyperbolic(hyp_encoder, hyp_decoder, train_dataloader, optimizer, criterion, track_energy=True)
            hyp_train.append(h_loss)
            layer_energy_epochwise.append(epoch_energies)

            h_test_loss, _, _ = evaluate_loss(hyp_encoder, hyp_decoder, test_dataloader, is_hyperbolic=True)
            hyp_test.append(h_test_loss)
            
            # Print info
            print(f"Epoch {epoch+1}/{epochs}: Hyperbolic Train={h_loss:.4f} Test={h_test_loss:.4f}")
            # Print energy 
            formatted_energies = [f"{e:.4f}" for e in epoch_energies]
            print(f"  Epoch {epoch}: {formatted_energies}")
            # Early stopping
            if len(hyp_test) >= early_stop_window + 1 :
                last_hyp_test = hyp_test[-1]
                prev_hyp_window = hyp_test[-early_stop_window-1:-1]

                if all(last_hyp_test >= prev for prev in prev_hyp_window):
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    hyp_early_stopped = True
                    for i in range (epoch+1, 100):
                        hyp_test.append(last_hyp_test) 
                        hyp_train.append(hyp_train[-1])
                        # Keep the last energy values for remaining epochs
                        layer_energy_epochwise.append(epoch_energies)
                    break
        if not hyp_early_stopped:
            print(f"\n✅ Training Hyperbolic Autoencoder completed all {epochs} epochs without early stopping.")
        


        # Save
        key_e = f"Experiment {exp+1}_Euclidean_lr{lr}_layers{'-'.join(map(str,layers))}"
        key_h = f"Experiment {exp+1}_Hyperbolic_lr{lr}_layers{'-'.join(map(str,layers))}"
        all_train_losses[key_e] = eucl_train
        all_test_losses[key_e] = eucl_test
        all_train_losses[key_h] = hyp_train
        all_test_losses[key_h] = hyp_test
        all_layer_energies[key_h] = layer_energy_epochwise  # Save energy tracking

        # Visualize reconstructions
        _, e_recon, orig = evaluate_loss(eucl_encoder, eucl_decoder, test_dataloader)
        _, h_recon, _ = evaluate_loss(hyp_encoder, hyp_decoder, test_dataloader, is_hyperbolic=True)
        visualize_reconstructions(orig[:10], e_recon[:10], h_recon[:10], lr, f"{layers}", eucl_test[-1], hyp_test[-1])
        import csv

        # Create the file
        filename = results_folder/f"Experiment_{exp+1}_losses_lr{lr}_layers{'-'.join(map(str, layers))}.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Euclidean Train Loss", "Hyperbolic Train Loss", "Euclidean Test Loss", "Hyperbolic Test Loss"])
            for et, ht, eTe, hTe in zip(eucl_train, hyp_train, eucl_test, hyp_test):
                writer.writerow([et, ht, eTe, hTe])
        
        # Save energy tracking data
        energy_filename = results_folder/f"Experiment_{exp+1}_energies_lr{lr}_layers{'-'.join(map(str, layers))}.csv"
        with open(energy_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Create header for energy columns
            num_layers = len(layer_energy_epochwise[0]) if layer_energy_epochwise else 0
            header = [f"Layer_{i+1}_Energy" for i in range(num_layers)]
            writer.writerow(header)
            
            # Write energy data for each epoch
            for epoch_energies in layer_energy_epochwise:
                writer.writerow(epoch_energies)
        
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
plt.savefig(results_folder/"train_loss_curves.png")
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
plt.savefig(results_folder/"test_loss_curves.png")
plt.show()

# Plot energy curves for hyperbolic models
plt.figure(figsize=(15, 10))
for key, energies in all_layer_energies.items():
    if energies:  # Check if we have energy data
        num_layers = len(energies[0])
        for layer_idx in range(num_layers):
            layer_energies = [epoch_energies[layer_idx] for epoch_energies in energies]
            plt.plot(range(1, len(layer_energies)+1), layer_energies, 
                    label=f"{key}_Layer_{layer_idx+1}", alpha=0.7)

plt.title("Layer-wise Activation Energy (Squared Sum of First Coordinate)")
plt.xlabel("Epoch")
plt.ylabel("Average Energy per Layer")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(results_folder/"layer_energy_curves.png", bbox_inches='tight')
plt.show()

# Plot average energy across all layers per experiment
plt.figure(figsize=(12, 6))
for key, energies in all_layer_energies.items():
    if energies:
        # Calculate average energy across all layers for each epoch
        avg_energies = [sum(epoch_energies)/len(epoch_energies) for epoch_energies in energies]
        plt.plot(range(1, len(avg_energies)+1), avg_energies, label=key, linewidth=2)

plt.title("Average Activation Energy Across All Layers")
plt.xlabel("Epoch")
plt.ylabel("Average Energy")
plt.legend()
plt.grid(True)
plt.savefig(results_folder/"average_energy_curves.png")
plt.show()
