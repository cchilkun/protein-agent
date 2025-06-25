""" [TODO] """

from torch.utils.data import DataLoader
from UTILS_postprocess import *

from _dataset import *
from _policy import *

from sklearn.model_selection import train_test_split

import numpy as np
import torch

print()
print("getting the workflow going...")

# ---------------------------------------------------------------------------------------------------- #
# Load trajectory paths
CONFIG_1_path = "/scratch/m000150/outputs_1000R_HSA_IBU/HSA_IBU_CONFIG_1_trajectory.pkl"
CONFIG_1 = load_trajectory(CONFIG_1_path)

train_data, test_data = train_test_split(CONFIG_1, test_size=0.2, random_state=42)
print("loaded the data...")

# ---------------------------------------------------------------------------------------------------- #
# Define residue vocabulary
protein_includes_list = [
    "ARG", "HIS", "LYS", "ASP", "GLU", "SER", "THR", "ASN", "GLN", 
    "GLY", "PRO", "CYS", "SEC", "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP"
]
residue_vocab = {res: i for i, res in enumerate(protein_includes_list)}
residue_vocab["UNK"] = len(residue_vocab)

# ---------------------------------------------------------------------------------------------------- #
# Create datasets
train_dataset = TrajectoryDataset(train_data, residue_vocab)
test_dataset = TrajectoryDataset(test_data, residue_vocab)

# ---------------------------------------------------------------------------------------------------- #
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}\n")

# ---------------------------------------------------------------------------------------------------- #
# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------------------------------------------------- #
# Model, optimizer, loss
#model = GNNPolicy(residue_vocab_size=len(residue_vocab), action_dim=3).to(device)
#optimizer = torch.optim.AdamW(model.parameters(), lr=1E-3)
#criterion = torch.nn.MSELoss()

diffusion_steps = 1000
model = DiffusionPolicy(residue_vocab_size=len(residue_vocab), action_dim=3, diffusion_steps=diffusion_steps).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1E-3)
criterion = torch.nn.MSELoss()

print(f"defined model -->\n{model}\n")
print(f"defined optimizer and criterion -->\n{optimizer}\n{criterion}\n")

# ---------------------------------------------------------------------------------------------------- #
# Training Loop with Evaluation
for epoch in range(50):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        protein_t = batch["protein_t0"].to(device)
        molecule_t = batch["micromolecule_t0"].to(device)
        true_action_t = batch["action_t0"].to(device).view(-1, 3)

        # ----------------- Sample random timestep ----------------- #
        batch_size = protein_t.batch.max().item() + 1
        t = torch.randint(0, diffusion_steps, (batch_size,), device=device)
        t_per_node = t[molecule_t.batch]  # (num_molecule_nodes,)

        # ----------------- Generate noisy actions ----------------- #
        noise = torch.randn_like(true_action_t)

        sqrt_alpha = model.sqrt_alphas_cumprod[t_per_node][:, None]  # (num_molecule_nodes, 1)
        sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[t_per_node][:, None]

        noisy_action_t = sqrt_alpha * true_action_t + sqrt_one_minus_alpha * noise

        # ----------------- Forward pass ----------------- #
        pred_noise = model(protein_t, molecule_t, t, noisy_action_t)

        # Loss = MSE between true noise and predicted noise
        loss = criterion(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        print(f"\tTrain Minibatch Loss = {loss.item():.4f}")

    print(f"[Epoch {epoch}] Train Loss = {total_train_loss:.4f}")

    # --------------------------------- Evaluation --------------------------------- #
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            protein_t = batch["protein_t0"].to(device)
            molecule_t = batch["micromolecule_t0"].to(device)
            true_action_t = batch["action_t0"].to(device).view(-1, 3)

            # Sample random timestep for eval too
            batch_size = protein_t.batch.max().item() + 1
            t = torch.randint(0, diffusion_steps, (batch_size,), device=device)
            t_per_node = t[molecule_t.batch]

            # Add noise
            noise = torch.randn_like(true_action_t)
            sqrt_alpha = model.sqrt_alphas_cumprod[t_per_node][:, None]
            sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[t_per_node][:, None]
            noisy_action_t = sqrt_alpha * true_action_t + sqrt_one_minus_alpha * noise

            # Forward pass
            pred_noise = model(protein_t, molecule_t, t, noisy_action_t)

            loss = criterion(pred_noise, noise)
            total_test_loss += loss.item()

    print(f"[Epoch {epoch}] Eval Loss = {total_test_loss:.4f}\n")

"""

# ---------------------------------------------------------------------------------------------------- #
# Training Loop with Evaluation
for epoch in range(50):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        protein_t = batch["protein_t0"].to(device)
        molecule_t = batch["micromolecule_t0"].to(device)
        true_action_t = batch["action_t0"].to(device).view(-1, 3)

        pred_action_t = model(protein_t, molecule_t)

        loss = criterion(pred_action_t, true_action_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        print(f"\tTrain Minibatch Loss = {loss.item():.4f}")

    print(f"[Epoch {epoch}] Train Loss = {total_train_loss:.4f}")

    # --------------------------------- Evaluation --------------------------------- #
    model.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            protein_t = batch["protein_t0"].to(device)
            molecule_t = batch["micromolecule_t0"].to(device)
            true_action_t = batch["action_t0"].to(device).view(-1, 3)

            pred_action_t = model(protein_t, molecule_t)
            loss = criterion(pred_action_t, true_action_t)
            total_test_loss += loss.item()

    print(f"[Epoch {epoch}] Eval Loss = {total_test_loss:.4f}\n")
"""
