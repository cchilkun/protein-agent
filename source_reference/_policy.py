"""
[TODO]

Author: Chetan Chilkunda
Date Created: 4 June 2025
Date Modified: 9 June 2025
"""

# **************************************************************************************************** #

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# **************************************************************************************************** #

# ---------------------------------------------------------
# Diffusion utilities
# ---------------------------------------------------------

def cosine_beta_schedule(timesteps, s=0.008):
    import numpy as np
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0.0001, 0.9999)

# ---------------------------------------------------------
# DiffusionPolicy model
# ---------------------------------------------------------

class DiffusionPolicy(nn.Module):
    def __init__(self, residue_vocab_size, hidden_dim=128, action_dim=3, diffusion_steps=1000):
        super().__init__()

        self.diffusion_steps = diffusion_steps

        # diffusion schedule
        betas = cosine_beta_schedule(diffusion_steps)
        self.register_buffer('betas', torch.tensor(betas, dtype=torch.float32))
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # GCN for protein graph → global embedding
        self.protein_gcn = nn.ModuleList([
            GCNConv(residue_vocab_size + 3, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # GCN for micromolecule graph → per-node embedding
        self.molecule_gcn = nn.ModuleList([
            GCNConv(residue_vocab_size + 3, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # timestep embedding MLP
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # noise prediction MLP head
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, protein_t, molecule_t, t, noisy_action):
        """
        Args:
            protein_t: protein graph batch
            molecule_t: micromolecule graph batch
            t: diffusion timesteps (batch_size,)
            noisy_action: noisy actions (num_molecule_nodes, action_dim)
        Returns:
            predicted noise (num_molecule_nodes, action_dim)
        """

        # Protein GCN → global embedding
        x_p_residues = F.one_hot(protein_t.residue_name, num_classes=self.protein_gcn[0].in_channels - 3).float()
        x_p = torch.cat([x_p_residues, protein_t.pos], dim=1)

        for conv in self.protein_gcn:
            x_p = conv(x_p, protein_t.edge_index)
            x_p = F.relu(x_p)

        protein_embedding = global_mean_pool(x_p, protein_t.batch)  # (batch_size, hidden_dim)

        # Molecule GCN → per-node embedding
        x_m_residues = F.one_hot(molecule_t.residue_name, num_classes=self.molecule_gcn[0].in_channels - 3).float()
        x_m = torch.cat([x_m_residues, molecule_t.pos], dim=1)

        for conv in self.molecule_gcn:
            x_m = conv(x_m, molecule_t.edge_index)
            x_m = F.relu(x_m)

        # Timestep embedding
        t_emb = self.timestep_embedding(t[:, None].float())  # (batch_size, hidden_dim)
        t_emb_per_node = t_emb[molecule_t.batch]  # (num_molecule_nodes, hidden_dim)

        # Repeat protein embedding per node
        protein_emb_per_node = protein_embedding[molecule_t.batch]  # (num_molecule_nodes, hidden_dim)

        # Concatenate all inputs to noise predictor
        input_features = torch.cat([
            x_m,                      # molecule node embedding
            protein_emb_per_node,     # protein context
            t_emb_per_node,           # timestep embedding
            noisy_action              # current noisy action
        ], dim=1)

        # Predict noise
        predicted_noise = self.noise_predictor(input_features)

        return predicted_noise


class GNNPolicy(torch.nn.Module):
    def __init__(self, residue_vocab_size, hidden_dim=128, action_dim=3):
        super().__init__()

        # defining the protein GNN
        self.protein_GNN = torch.nn.ModuleList([
            GCNConv(residue_vocab_size+3, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # defining the micromolecule GNN
        self.micromolecule_GNN = torch.nn.ModuleList([
            GCNConv(residue_vocab_size+3, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # defining the final MLP to predict action transitions
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            #torch.nn.Linear(hidden_dim, hidden_dim // 2),
            #torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, protein_t, micromolecule_t):
        ''' process protein as before, for context '''
        x_p_residues = torch.nn.functional.one_hot(protein_t.residue_name, num_classes=self.protein_GNN[0].in_channels - 3).float()
        x_p = torch.cat([x_p_residues, protein_t.pos], dim=1)

        for conv_layer in self.protein_GNN:
            x_p = conv_layer(x_p, protein_t.edge_index)
            x_p = torch.nn.functional.relu(x_p)

        protein_embedding = global_mean_pool(x_p, protein_t.batch)  # shape [batch, hidden_dim]

        ''' process micromolecule nodewise '''
        x_m_residues = torch.nn.functional.one_hot(micromolecule_t.residue_name, num_classes=self.micromolecule_GNN[0].in_channels - 3).float()
        x_m = torch.cat([x_m_residues, micromolecule_t.pos], dim=1)

        for conv_layer in self.micromolecule_GNN:
            x_m = conv_layer(x_m, micromolecule_t.edge_index)
            x_m = torch.nn.functional.relu(x_m)

        ''' inject protein context to each micromolecule node '''
        # repeat protein embedding to match micromolecule nodes
        protein_context = protein_embedding[micromolecule_t.batch]  # [num_molecule_nodes, hidden_dim]

        # concatenate per-node micromolecule features with global protein context
        system_node_features = torch.cat([x_m, protein_context], dim=1)  # [num_molecule_nodes, 2*hidden_dim]

        ''' predict per-node displacements '''
        per_node_action = self.MLP(system_node_features)  # [num_molecule_nodes, 3]

        return per_node_action

# **************************************************************************************************** #
