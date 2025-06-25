"""
This file contains utilities relevant for postprocessing the collected temporal molecular dynamics simulation data (ran via the OpenMM framework).

Author: Chetan Chilkunda
Date Created: 26 May 2025
Date Modified: 3 June 2025
"""

# **************************************************************************************************** #

import os
import numpy as np

import pickle
import mdtraj as md
import networkx as nx
from tqdm import tqdm

from openmm.app import PDBFile, Topology
from openmm.unit import angstrom

# ************************************* POSTPROCESSING UTILITIES ************************************* #

def filter_directory_by_REGEX(directory_path, regex_pattern, start_index, end_index):
    """
    Processes a simulation directory of PDB files by the given REGEX pattern and filters the files by their numeric
    index to be within [start_index, end_index] (inclusive), and returns the filtered list of REGEX matches (as file paths).

    - @NOTE: the PDB files are assumed to have format "<FILENAME>_<INDEX>.xyz" as specified by the regex_pattern

    Inputs:
    - @param[in] directory_path (str) : path to the simulation directory
    - @param[in] regex_pattern (REGEX pattern) : REGEX pattern by which to filter the directory over
    - @param[in] start_index : starting index for filtering the given directory files (inclusive)
    - @param[in] end_index : ending index for filtering the given directory files (inclusive)

    Outputs:
    - @return (list) : list of strings corresponding to the filtered simulation file paths found via REGEX matches 
    """

    assert(start_index < end_index) # safety check

    regex_match_paths = []
    for file in os.listdir(directory_path):
        match = regex_pattern.match(file)

        if (match):
            checkpoint_index = int(match.group(1))
            if (start_index <= checkpoint_index <= end_index): regex_match_paths.append(os.path.join(directory_path, file))

    return regex_match_paths

def postprocess_PDB_checkpoint(checkpoint_topology, checkpoint_positions, includes_list):
    """
    Filters a given OpenMM Topology and its corresponding atomic positions to only include residues whose names are 
    specified in 'includes_list'. Constructs and returns a new topology and new atomic positions array containing only the selected atoms.

    - @NOTE: assumes the strings found in includes_list are valid residue names (found in the pdb_checkpoint_file input)

    Inputs:
    - @param[in] checkpoint_topology (OpenMM Topology) : OpenMM Topology object from a checkpoint PDB file
    - @param[in] checkpoint_positions (numpy.ndarray) : atomic positions corresponding to the specified topology
    - @param[in] includes_list (list) : list of strings of residue names to retain

    Outputs:
    - @return (tuple) : tuple containing the (1) postprocessed checkpoint's topology (Topology object) and (2) its atomic positions (numpy.ndarray)
    """

    processed_topology, processed_positions = Topology(), []
    processed_chain = processed_topology.addChain() # adding a single chain for writing the desired simulation system object
    atom_mapping = {}

    for chain in checkpoint_topology.chains():
        for residue in chain.residues():
            if (residue.name in includes_list):
                processed_residue = processed_topology.addResidue(residue.name, processed_chain, residue.id)
                for atom in residue.atoms():
                    processed_atom = processed_topology.addAtom(atom.name, atom.element, processed_residue)
                    atom_mapping[atom] = processed_atom
                    processed_positions.append(checkpoint_positions[atom.index])

    # adding the bonds that correspond to the residues in the includes_list parameter
    for atom_1, atom_2 in checkpoint_topology.bonds():
        if (atom_1 in atom_mapping and atom_2 in atom_mapping):
            processed_topology.addBond(atom_mapping[atom_1], atom_mapping[atom_2])

    return (processed_topology, np.array(processed_positions))

def create_topology_template(topology):
    """
    Converts an OpenMM Topology object to a NetworkX graph structure without any positional information, and 
    returns both the graph template and the atom mapping (OpenMM Atom : index) for downstream post-processing.

    Inputs:
    - @param[in] topology (OpenMM Topology) : the topology (connectivity information) associated with an OpenMM object

    Outputs:
    - @return (tuple) : (NetworkX graph, atom mapping dictionary)
    """

    G = nx.Graph() # instantiating a NetworkX graph

    atom_mapping = {}
    # adding the atoms as graph nodes
    for i, atom in enumerate(topology.atoms()):
        atom_mapping[atom] = i
        G.add_node(i, element_symbol=atom.element.symbol, residue_name=atom.residue.name) # not encoding any atomic positional information

    # adding bonds as graph edges (node connectivities)
    for atom_1, atom_2 in topology.bonds():
        if (atom_1 in atom_mapping and atom_2 in atom_mapping):
            G.add_edge(atom_mapping[atom_1], atom_mapping[atom_2])

    return (G, atom_mapping)

def instantiate_graph_with_positions(G_template, positions):
    """
    Clones the given graph template and updates every node's attributes to include positional information.

    - @NOTE: assumes that the node indices in G_template correspond directly to the indices in the positions array

    Inputs:
    - @param[in] G_template (NetworkX Graph) : graph with static topology (nodes, edges) information and node metadata, 
    but without any node-specific positional information.
    - @param[in] positions (np.ndarray) : specific atomic positions corresponding to the given topology

    Outputs:
    - @return (NetworkX Graph) : a deep copy of the input graph template with each node updated with positional information
    """

    G = G_template.copy(as_view=False)

    # for node_index in G.nodes:
    #     G.nodes[node_index]["x"] = positions[node_index][0]
    #     G.nodes[node_index]["y"] = positions[node_index][1]
    #     G.nodes[node_index]["z"] = positions[node_index][2]

    pos_array = positions
    nx.set_node_attributes(G, {i: {"x": pos_array[i][0], "y": pos_array[i][1], "z": pos_array[i][2]} for i in range(len(pos_array))})

    return G

def extract_simulation_trajectory(simulation_files, protein_includes_list, micromolecule_includes_list):
    """
    Postprocesses a list of PDB checkpoint files (from a simulation) into a trajectory of graph-based state transitions. This function 
    processes consecutive pairs of molecular dynamics (MD) checkpoint frames into graph-based representations for both the protein and 
    micromolecule components for each timestep. It also computes the atomic displacement of the micromolecule between frames as the 'action'.

    - @NOTE: the generated trajectory is a 9-tuple representing the following:
    (curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,    # current state
     next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,    # next state
     action)                                                                                        # action (transition) from current to next state

     - @NOTE: assuming that the simulation_files input are ordered temporally from a simulation

     - @NOTE: the graphs are instantiated via fence-posting for efficiency, and assumes that the atom connectivity (topology) is static

     Inputs:
     - @param[in] simulation_files (list) : a sorted list of PDB checkpoint file paths
     - @param[in] protein_includes_list (list) : list of strings of residue names to retain for the protein
     - @param[in] micromolecule_includes_list (list) : list of strings of residue names to retain for the micromolecule

     Outputs:
     - @return (list) : list of 9-tuples corresponding to the simulation trajectory as (current state, next state, action) transitions
    """

    simulation_trajectory = []

    # fence-posting the iteration through simulation checkpoint frames
    curr_checkpoint_file = PDBFile(simulation_files[0])
    curr_checkpoint_topology = curr_checkpoint_file.getTopology()
    curr_checkpoint_positions = np.array([pos.value_in_unit(angstrom) for pos in curr_checkpoint_file.positions])

    # pre-computing the graph templates for both the protein and micromolecule (as atom connectivity is assumed as static)
    curr_protein_topology, curr_protein_positions = postprocess_PDB_checkpoint(curr_checkpoint_topology, curr_checkpoint_positions, protein_includes_list)
    protein_G_template, _ = create_topology_template(curr_protein_topology)
    curr_protein_G = instantiate_graph_with_positions(protein_G_template, curr_protein_positions)

    curr_micromolecule_topology, curr_micromolecule_positions = postprocess_PDB_checkpoint(curr_checkpoint_topology, curr_checkpoint_positions, micromolecule_includes_list)
    micromolecule_G_template, _ = create_topology_template(curr_micromolecule_topology)
    curr_micromolecule_G = instantiate_graph_with_positions(micromolecule_G_template, curr_micromolecule_positions)

    for i in tqdm(range(1, len(simulation_files)), ascii=True, desc="postprocessing checkpoints"): # post-processing the simulation checkpoints into RL inputs (state, action) trajectories
        # getting the 'next' simulation checkpoint frame
        next_checkpoint_file = PDBFile(simulation_files[i])
        next_checkpoint_topology = next_checkpoint_file.getTopology()
        next_checkpoint_positions = np.array([pos.value_in_unit(angstrom) for pos in next_checkpoint_file.positions])

        # instantiating the protein's 'next' state graph representation via the pre-computed template
        _, next_protein_positions = postprocess_PDB_checkpoint(next_checkpoint_topology, next_checkpoint_positions, protein_includes_list)
        next_protein_G = instantiate_graph_with_positions(protein_G_template, next_protein_positions)

        # instantiating the micromolecule's 'next' state graph representation via the pre-computed template
        _, next_micromolecule_positions = postprocess_PDB_checkpoint(next_checkpoint_topology, next_checkpoint_positions, micromolecule_includes_list)
        next_micromolecule_G = instantiate_graph_with_positions(micromolecule_G_template, next_micromolecule_positions)

        # defining the action transition as atomic displacement of the micromolecule (δ micromolecule positions from current state to next state)
        # @NOTE: this defines the agent as a 'protein agent' that predicts the micromolecule's trajectory
        action = next_micromolecule_positions - curr_micromolecule_positions

        simulation_trajectory.append((
            curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,
            next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions, 
            action
        ))

        # updating the fenceposting as 'next' state is the future's 'current' state
        curr_protein_G = next_protein_G
        curr_protein_positions = next_protein_positions
        curr_micromolecule_G = next_micromolecule_G
        curr_micromolecule_positions = next_micromolecule_positions

    return simulation_trajectory

def save_trajectory(trajectory_data, trajectory_pkl_path):
    """
    Serializes and saves a molecular simulation trajectory to a .pkl file. The trajectory is a list 
    of graph-based state-action transitions generated during postprocessing of molecular dynamics (MD) simulations.

    - @NOTE: the saved trajectory is a list of 9-tuples representing the following:
    (curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,    # current state
     next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,    # next state
     action)                                                                                        # action (transition) from current to next state

    Inputs:
    - @param[in] trajectory_data (list) : list of 9-tuples representing (current state, next state, action)
    - @param[in] trajectory_pkl_path (str) : path to the destination .pkl file where the trajectory will be saved

    Outputs:
    - @return None : the trajectory is written to memory (disk), no return value
    """

    # saving the collected trajectory via pickle
    with open(trajectory_pkl_path, "wb") as f:
        pickle.dump(trajectory_data, f)

def load_trajectory(trajectory_pkl_path):
    """
    Loads a serialized molecular simulation trajectory from a .pkl file. The expected trajectory data format 
    is a list of graph-based state-action transitions generated during postprocessing of molecular dynamics (MD) simulations.

    - @NOTE: the loaded trajectory is a 9-tuple representing the following:
    (curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,    # current state
     next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,    # next state
     action)                                                                                        # action (transition) from current to next state

    Inputs:
    - @param[in] trajectory_pkl_path (str) : path to the saved .pkl file containing the simulation trajectory

    Outputs:
    - @return (list) : deserialized trajectory data as a list of 9-tuples representing (current state, next state, action)
    """

    with open(trajectory_pkl_path, "rb") as f:
        trajectory_data = pickle.load(f)

    return trajectory_data

# **************************************************************************************************** #


def extract_simulation_trajectory_fast(simulation_files, protein_includes_list, micromolecule_includes_list):
    """
    Optimized version of extract_simulation_trajectory:
    - Uses mdtraj to batch load all positions (avoids slow PDB parsing)
    - Parses topology once
    - Avoids redundant graph rebuilding
    - Much faster overall
    """

    # Load the full trajectory (positions + topology once)
    print("[INFO] Loading trajectory with mdtraj...")
    traj = md.load(simulation_files, top=simulation_files[0])  # loads all frames at once!

    # Extract atom selections (indices) once
    all_atom_names = [atom.name for atom in traj.topology.atoms]
    all_residue_names = [atom.residue.name for atom in traj.topology.atoms]

    # Create selection masks
    protein_atom_indices = np.array([
        i for i, resname in enumerate(all_residue_names) if resname in protein_includes_list
    ])
    micromolecule_atom_indices = np.array([
        i for i, resname in enumerate(all_residue_names) if resname in micromolecule_includes_list
    ])

    print(f"[INFO] Protein atoms selected: {len(protein_atom_indices)}")
    print(f"[INFO] Micromolecule atoms selected: {len(micromolecule_atom_indices)}")

    # Build static graph templates once
    print("[INFO] Building graph templates...")
    # @NOTE: mdtraj topology is compatible with OpenMM Topology
    protein_topology = traj.top.to_openmm()
    micromolecule_topology = traj.top.to_openmm()

    # Postprocess only ONCE to build topology and graph template
    protein_topology_processed, _ = postprocess_PDB_checkpoint(protein_topology, np.zeros((traj.n_atoms, 3)), protein_includes_list)
    micromolecule_topology_processed, _ = postprocess_PDB_checkpoint(micromolecule_topology, np.zeros((traj.n_atoms, 3)), micromolecule_includes_list)

    protein_G_template, _ = create_topology_template(protein_topology_processed)
    micromolecule_G_template, _ = create_topology_template(micromolecule_topology_processed)

    # Iterate over trajectory frames as pairs
    print("[INFO] Processing trajectory frames...")
    simulation_trajectory = []
    for i in tqdm(range(traj.n_frames - 1), ascii=True, desc="postprocessing trajectory"):
        # Current frame positions
        curr_positions = traj.xyz[i] * 10.0  # convert nm → angstrom
        next_positions = traj.xyz[i + 1] * 10.0

        # Select atom positions
        curr_protein_positions = curr_positions[protein_atom_indices]
        curr_micromolecule_positions = curr_positions[micromolecule_atom_indices]

        next_protein_positions = next_positions[protein_atom_indices]
        next_micromolecule_positions = next_positions[micromolecule_atom_indices]

        # Build graphs
        curr_protein_G = instantiate_graph_with_positions(protein_G_template, curr_protein_positions)
        next_protein_G = instantiate_graph_with_positions(protein_G_template, next_protein_positions)

        curr_micromolecule_G = instantiate_graph_with_positions(micromolecule_G_template, curr_micromolecule_positions)
        next_micromolecule_G = instantiate_graph_with_positions(micromolecule_G_template, next_micromolecule_positions)

        # Compute action (micromolecule displacement)
        action = next_micromolecule_positions - curr_micromolecule_positions

        # Store tuple
        simulation_trajectory.append((
            curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,
            next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,
            action
        ))

    return simulation_trajectory


def extract_simulation_trajectory_ultrafast(simulation_files, protein_includes_list, micromolecule_includes_list):
    """
    Ultimate fast version:
    - Load trajectory once
    - Parse topology once
    - Precompute atom indices once
    - Reuse graph templates
    - Avoid all per-frame heavy logic
    """

    # Load the full trajectory (positions + topology once)
    print("[INFO] Loading trajectory with mdtraj...")
    traj = md.load(simulation_files, top=simulation_files[0])  # loads all frames at once!

    # Precompute selections
    all_atom_names = [atom.name for atom in traj.topology.atoms]
    all_residue_names = [atom.residue.name for atom in traj.topology.atoms]

    protein_atom_indices = np.array([
        i for i, resname in enumerate(all_residue_names) if resname in protein_includes_list
    ])
    micromolecule_atom_indices = np.array([
        i for i, resname in enumerate(all_residue_names) if resname in micromolecule_includes_list
    ])

    print(f"[INFO] Protein atoms selected: {len(protein_atom_indices)}")
    print(f"[INFO] Micromolecule atoms selected: {len(micromolecule_atom_indices)}")

    # Build static graph templates ONCE
    print("[INFO] Building graph templates...")
    def create_graph_template_from_indices(atom_indices):
        G = nx.Graph()
        for i, atom_idx in enumerate(atom_indices):
            atom = traj.topology.atom(atom_idx)
            G.add_node(i, element_symbol=atom.element.symbol, residue_name=atom.residue.name)
        # Add edges based on original topology bonds
        for bond in traj.topology.bonds:
            if bond.atom1.index in atom_indices and bond.atom2.index in atom_indices:
                idx1 = np.where(atom_indices == bond.atom1.index)[0][0]
                idx2 = np.where(atom_indices == bond.atom2.index)[0][0]
                G.add_edge(idx1, idx2)
        return G

    protein_G_template = create_graph_template_from_indices(protein_atom_indices)
    micromolecule_G_template = create_graph_template_from_indices(micromolecule_atom_indices)

    # Main loop → process pairs of frames
    print("[INFO] Processing trajectory frames...")
    simulation_trajectory = []
    for i in tqdm(range(traj.n_frames - 1), ascii=True, desc="postprocessing trajectory"):
        curr_positions = traj.xyz[i] * 10.0  # nm → angstrom
        next_positions = traj.xyz[i + 1] * 10.0

        curr_protein_positions = curr_positions[protein_atom_indices]
        curr_micromolecule_positions = curr_positions[micromolecule_atom_indices]

        next_protein_positions = next_positions[protein_atom_indices]
        next_micromolecule_positions = next_positions[micromolecule_atom_indices]

        # Build graphs by inserting positions into templates
        def instantiate_graph(G_template, positions):
            G = G_template.copy(as_view=False)
            nx.set_node_attributes(G, {i: {"x": positions[i][0], "y": positions[i][1], "z": positions[i][2]} for i in range(len(positions))})
            return G

        curr_protein_G = instantiate_graph(protein_G_template, curr_protein_positions)
        next_protein_G = instantiate_graph(protein_G_template, next_protein_positions)

        curr_micromolecule_G = instantiate_graph(micromolecule_G_template, curr_micromolecule_positions)
        next_micromolecule_G = instantiate_graph(micromolecule_G_template, next_micromolecule_positions)

        # Action is micromolecule displacement
        action = next_micromolecule_positions - curr_micromolecule_positions

        # Store transition tuple
        simulation_trajectory.append((
            curr_protein_G, curr_protein_positions, curr_micromolecule_G, curr_micromolecule_positions,
            next_protein_G, next_protein_positions, next_micromolecule_G, next_micromolecule_positions,
            action
        ))

    return simulation_trajectory


# ******************************** HELPER FUNCTIONS FOR VISUALIZATION ******************************** #

def create_trajectory_from_PDBs(pbd_paths_list, output_trajectory_path):
    """
    Concatenates len(pbd_paths_list)-many PDB files into a single PDB file for trajectory visualization.

    - @NOTE: writes the trajectory in the order given by the pdb_paths_list

    Inputs:
    - @param[in] pbd_paths_list (list) : list of input PDB paths (the files to concatenate into a trajectory)
    - @param[in] output_trajectory_path (str) : path to a PDB file to save the generated trajectory

    Outputs:
    - @write[out] output_trajectory_path : writes the trajectory (series of PDB states) to a PDB file saved at this location
    """

    with open(output_trajectory_path, "w") as f:
        for i, pdb_path in enumerate(pbd_paths_list):
            with open(pdb_path, "r") as simulation_state_pdb:
                f.write(f"MODEL     {i + 1}\n")

                for line in simulation_state_pdb:
                    # skipping the existing MODEL/ENDMDL/END lines to avoid duplicates
                    if (line.startswith("MODEL") or line.startswith("ENDMDL") or line.strip() == "END"): continue
                    f.write(line)
                    
                f.write("ENDMDL\n")

# **************************************************************************************************** #
