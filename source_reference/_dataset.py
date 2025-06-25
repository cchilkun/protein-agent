"""
[TODO]

Author: Chetan Chilkunda
Date Created: 3 June 2025
Date Modified: 4 June 2025
"""

# **************************************************************************************************** #

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

# **************************************************************************************************** #

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    The trajectory data is a 9-tuple representing the following:
    (curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,    # current state
     next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,    # next state
     action)                                                                                        # action (transition) from current to next state
    """

    def __init__(self, trajectory_data, residue_vocab):
        self.trajectory = trajectory_data
        self.residue_vocab = residue_vocab

    def __len__(self): return len(self.trajectory)

    def numerify_node_metadata(self, G):
        """
        Numerically encodes the node-level molecular metadata from a NetworkX graph into numerical tensor form. This function 
        converts string attributes of atoms, specifically the residue name, into integer indices using predefined vocabularies.

        - @NOTE: missing or unknown attribute values are replaced with the "UNK" token in the respective vocabulary

        Inputs:
        - @param[in] G (networkx.Graph) : a molecular graph where each node has the attribute 'residue_name'

        Outputs:
        - @return (torch.Tensor) : tensor of residue indices (dtype=torch.long)
        """

        residue_list = []
        for _, attr in G.nodes(data=True):
            residue = attr.get("residue_name", "UNK")
            residue_list.append(self.residue_vocab.get(residue, self.residue_vocab["UNK"]))

        residue_list_tensor = torch.tensor(residue_list, dtype=torch.long)

        return residue_list_tensor

    def __getitem__(self, idx):
        """ [TODO] """

        (protein_G_t0, protein_pos_t0, micromolecule_G_t0, micromolecule_pos_t0, 
         protein_G_t1, protein_pos_t1, micromolecule_G_t1, micromolecule_pos_t1, action_t) = \
            self.trajectory[idx]

        # converting the 'current' protein graph representation into a torch geometric object
        protein_data_t0 = from_networkx(protein_G_t0)
        protein_data_t0.pos = torch.tensor(protein_pos_t0, dtype=torch.float)
        protein_data_t0.residue_name = self.numerify_node_metadata(protein_G_t0)
        for attr in ["element_symbol", "x", "y", "z"]:
            if hasattr(protein_data_t0, attr): delattr(protein_data_t0, attr)
            
        # converting the 'current' micromolecule graph representation into a torch geometric object
        micromolecule_data_t0 = from_networkx(micromolecule_G_t0)
        micromolecule_data_t0.pos = torch.tensor(micromolecule_pos_t0, dtype=torch.float)
        micromolecule_data_t0.residue_name = self.numerify_node_metadata(micromolecule_G_t0)
        for attr in ["element_symbol", "x", "y", "z"]:
            if hasattr(micromolecule_data_t0, attr): delattr(micromolecule_data_t0, attr)

        # converting the 'next' protein graph representation into a torch geometric object
        protein_data_t1 = from_networkx(protein_G_t1)
        protein_data_t1.pos = torch.tensor(protein_pos_t1, dtype=torch.float)
        protein_data_t1.residue_name = self.numerify_node_metadata(protein_G_t1)
        for attr in ["element_symbol", "x", "y", "z"]:
            if hasattr(protein_data_t1, attr): delattr(protein_data_t1, attr)

        # converting the 'next' micromolecule graph representation into a torch geometric object
        micromolecule_data_t1 = from_networkx(micromolecule_G_t1)
        micromolecule_data_t1.pos = torch.tensor(micromolecule_pos_t1, dtype=torch.float)
        micromolecule_data_t1.residue_name = self.numerify_node_metadata(micromolecule_G_t1)
        for attr in ["element_symbol", "x", "y", "z"]:
            if hasattr(micromolecule_data_t1, attr): delattr(micromolecule_data_t1, attr)

        action_tensor = torch.tensor(action_t, dtype=torch.float)

        trajectory_frame = {
            "protein_t0"        : protein_data_t0,
            "micromolecule_t0"  : micromolecule_data_t0,
            "protein_t1"        : protein_data_t1,
            "micromolecule_t1"  : micromolecule_data_t1,
            "action_t0"         : action_tensor,
        }

        return trajectory_frame

def collate_fn(batch):
    """
    Collate function combines a list of individual samples (a batch) into a single batch object that can be fed 
    into a model during training or inference -- defining how to merge a list of dataset items returned by __getitem__ into a batch.
    """

    batched_data = {
        "protein_t0"            : Batch.from_data_list([item["protein_t0"] for item in batch]),
        "micromolecule_t0"      : Batch.from_data_list([item["micromolecule_t0"] for item in batch]),
        "protein_t1"            : Batch.from_data_list([item["protein_t1"] for item in batch]),
        "micromolecule_t1"      : Batch.from_data_list([item["micromolecule_t1"] for item in batch]),
        "action_t0"             : torch.stack([item["action_t0"] for item in batch]),
    }

    return batched_data

# **************************************************************************************************** #
