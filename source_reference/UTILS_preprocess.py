"""
This file contains utilities for preprocessing a protein PDB file and a micromolecule SMILES string for temporal molecular dynamics simulations. 

Author: Chetan Chilkunda
Date Created: 8 May 2025
Date Modified: 2 June 2025
"""

# **************************************************************************************************** #

import random
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pdbfixer import PDBFixer # needed for cleaning / fixing the protein PDB file

from openmm.app import PDBFile
from openmm.unit import angstrom, dalton

# importing rdkit module for reading and processing the SMILES string
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit.topology import Molecule # needed for generating the micromolecule's conformational isomers

# ***************************************** HELPER FUNCTIONS ***************************************** #

# TODO: add protein sequence truncation by sequence index?
def preprocess_protein(protein_pdb_path, output_protein_pdb_path, pH=7.4):
    """
    Cleans, rebuilds, and completes the protein PDB file by adding missing residues, atoms, and hydrogens.
    
    - @NOTE: PDBFixer documentation: https://htmlpreview.github.io/?https://github.com/openmm/pdbfixer/blob/master/Manual.html

    Inputs:
    - @param[in] protein_pdb_path (str) : path to the input protein PDB file
    - @param[in] output_protein_pdb_path (str) : path to a PDB file to save the updated protein structure
    - @param[in] pH (float) : pH value for the protein's protonation state (default is biological pH -- 7.4)

    Outputs:
    - @write[out] output_protein_pdb_path : writes the preprocessed protein structure to a PDB file saved at this location
    """

    fixer = PDBFixer(filename=protein_pdb_path)

    fixer.findMissingResidues()
    # only keeping missing residues if they are in the middle of a chain, and getting rid of entries that are at the start and end of the chain
    # @NOTE: this code is adapted from the PDBFixer documentation (linked above)
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in keys:
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            fixer.missingResidues[key] = []

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.removeHeterogens(True) # removing all HETATM atoms, excluding crystallographic water molecules

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=pH)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_protein_pdb_path, "w"))

def sample_bounding_sphere(pdb_path, scaling_factor):
    """
    Calculates the minimum bounding sphere that encloses the given PDB structure, scales this sphere (as specified), and 
    returns a tuple of the sampled position within the bounding sphere, the center point of the sphere, and the sphere's radius.

    - @NOTE: uniformly samples a position within the bounding sphere

    - @NOTE: returning the center point and radius of the sphere for visualization purposes

    Inputs:
    - @param[in] pdb_path (str) : path to the PDB file
    - @param[in] scaling_factor (float) : controllable factor to scale the MBS radius

    Outputs:
    - @return[out] (tuple) : a tuple of the sampled position (numpy.ndarray), 
        the sphere's center position (numpy.ndarray), and its radius (all in angstroms)
    """

    pdb_object = PDBFile(pdb_path)

    # extracting the structure's atomic positions and corresponding masses
    positions_array = np.array([pos.value_in_unit(angstrom) for pos in pdb_object.positions])
    masses_array = np.array([atom.element.mass.value_in_unit(dalton) for atom in pdb_object.topology.atoms()])

    # computing the center of mass (in angstroms)
    mbs_center_position = np.average(positions_array, axis=0, weights=masses_array)

    # computing the Euclidean distances from the computed center of mass to all atoms
    euclidean_distances = np.linalg.norm(positions_array - mbs_center_position, axis=1)

    # computing the radius of the minimum bounding sphere (MBS) that encloses the structure
    # @NOTE: modifying / scaling the computing MBS radius for greater diversity in sampling initial positions
    bs_radius = scaling_factor * np.max(euclidean_distances)

    # uniformly sampling a random point inside the sphere (sampling in 3D space with mean=0 and stdev=1)
    random_position = np.random.normal(0, 1, 3)
    unit_direction = random_position / np.linalg.norm(random_position)
    sampled_radius = bs_radius * np.cbrt(np.random.uniform(0, 1)) # cube root to enforce uniform sampling within the sphere
    sampled_position = mbs_center_position + unit_direction * sampled_radius

    return sampled_position, mbs_center_position, bs_radius

def instantiate_micromolecule(SMILES_string, desired_com_position):
    """
    Instantiates a micromolecule from the given SMILES string, places it at the specified center of mass position (with random rotation), 
    and returns the micromolecule's OpenMM topology object and its atomic positions.

    Inputs:
    - @param[in] SMILES_string (str) : the SMILES string of the micromolecule
    - @param[in] desired_com_position (numpy.ndarray) : the desired center of mass position (in angstroms) for the micromolecule

    Outputs:
    - @return[out] (tuple) : a tuple containing the micromolecule's OpenMM topology object, its atomic positions (in angstroms), and the OpenFF micromolecule's representation
    """

    rdkit_molecule_object = Chem.MolFromSmiles(SMILES_string)
    rdkit_molecule_object = Chem.AddHs(rdkit_molecule_object, explicitOnly=False) # adding both implicit and explicit hydrogens

    # using default ETKDGv3 algorithm (which accounts for torsion angle preferences from the Cambridge Structural Database) to clean up the structure
    AllChem.EmbedMolecule(rdkit_molecule_object)
    # using the Universal Force Field to optimize the molecular structure
    AllChem.UFFOptimizeMolecule(rdkit_molecule_object)

    # converting RDKit Molecule object to OpenFF Molecule object
    openff_molecule = Molecule.from_rdkit(rdkit_molecule_object, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    # generating the OpenFF 3D structure conformational isomers
    # @NOTE: n_conformers=10 to generate several molecule conformers (to get diversity in isomer structure)
    openff_molecule.generate_conformers(n_conformers=10)

    # assigning AM1-BCC charges to the micromolecule (highly accurate)
    openff_molecule.assign_partial_charges(partial_charge_method="am1bcc")

    # randomly selecting one of the generated conformers
    # @NOTE: random.randint() by default is inclusive of both endpoints
    random_index = random.randint(0, len(openff_molecule.conformers) - 1)

    micromolecule_topology = openff_molecule.to_topology().to_openmm()
    micromolecule_positions = openff_molecule.conformers[random_index].m_as("angstrom")

    ''' rotating and translating the micromolecule to the specified center of mass position '''

    # computing the micromolecule's current center of mass (in angstroms)
    masses_array = np.array([atom.element.mass.value_in_unit(dalton) for atom in micromolecule_topology.atoms()])
    current_com_position = np.average(micromolecule_positions, axis=0, weights=masses_array)

    # moving the micromolecule's center of mass to the coordinate system's origin (for symmetric rotation)
    micromolecule_positions -= current_com_position
    # arbitrarily rotating the micromolecule's atomic positions (for initial configuration diversity)
    random_rotation_matrix = Rotation.random().as_matrix()
    micromolecule_positions = micromolecule_positions @ np.transpose(random_rotation_matrix)
    # translating the micromolecule to the specified (new) center of mass position
    micromolecule_positions += desired_com_position

    return micromolecule_topology, micromolecule_positions, openff_molecule

def check_pairwise_steric_clashes(protein_pdb_path, micromolecule_positions, steric_clash_distance):
    """
    Checks if any of the micromolecule's atomic positions are within a given steric clash distance (exclusive end point) from the protein's structure.

    Inputs:
    - @param[in] protein_pdb_path (str) : path to the protein PDB file
    - @param[in] micromolecule_positions (numpy.ndarray) : the collection of atomic positions in angstroms with shape [NUM_ATOMS, 3]
    - @param[in] steric_clash_distance (float) : the threshold upper bound distance (in angstroms) that defines steric clashes

    Outputs:
    - @return[out] (bool) : True if there exists any steric clashes between the micromolecule and protein structure, and False otherwise
    """

    protein_pdb_object = PDBFile(protein_pdb_path)

    # extracting the protein structure's atomic positions (in angstroms)
    protein_positions = np.array([pos.value_in_unit(angstrom) for pos in protein_pdb_object.positions])

    # creating a KD-tree for efficient nearest neighbor search
    protein_structure_search_tree = cKDTree(protein_positions)

    # querying the protein's structure to identify any micromolecule atomic positions that are within the given steric clash distance
    distances, _ = protein_structure_search_tree.query(micromolecule_positions, distance_upper_bound=steric_clash_distance)

    return np.any(distances < steric_clash_distance)

def write_micromolecule_to_PDB(openmm_topology, micromolecule_positions, output_micromolecule_pdb_path):
    """
    Writes the micromolecule's OpenMM topology object and numpy.ndarray of its atomic positions to a PDB file.

    Inputs:
    - @param[in] openmm_topology (OpenMM Topology) : the micromolecule's corresponding OpenMM Topology object
    - @param[in] micromolecule_positions (numpy.ndarray) : the collection of atomic positions in angstroms with shape [NUM_ATOMS, 3]
    - @param[in] output_micromolecule_pdb_path (str) : path to the output PDB file to write the micromolecule's structure

    Outputs:
    - @write[out] output_micromolecule_pdb_path : writes the micromolecule's structure to a PDB file saved at this location
    """

    # converting numpy array of positions into an OpenMM positions array
    openmm_positions = micromolecule_positions * angstrom

    # writing the micromolecule's topology and positions to PDB format
    with open(output_micromolecule_pdb_path, "w") as f:
        PDBFile.writeFile(openmm_topology, openmm_positions, f)

# ******************************** HELPER FUNCTIONS FOR VISUALIZATION ******************************** #

def generate_spherical_shell(center_point, bounding_radius_angstroms, scaling_factor=1.0, num_points=100):
    """
    Generates random points on the surface of a sphere with a given radius about the given center point.

    Inputs:
    - @param[in] center_point (numpy.ndarray): the 3D coordinates of the center of the sphere (e.g., the COM of some molecular structure)
    - @param[in] bounding_radius_angstroms (float): the radius of the spherical shell (in Ångströms)
    - @param[in] scaling_factor (float): scaling factor to adjust the radius of the shell (default is 1.0)
    - @param[in] num_points (int): the number of points to generate on the shell surface

    Outputs:
    - @return (list of numpy.ndarray): list of num_points points on the spherical shell
    """

    points = []
    for _ in range(num_points):
        # generating random spherical coordinates
        theta = random.uniform(0, 2 * np.pi) # azimuthal angle
        phi = np.arccos(random.uniform(-1, 1)) # polar angle

        # converting the spherical coordinates to cartesian coordinates
        x = center_point[0] + scaling_factor * bounding_radius_angstroms * np.sin(phi) * np.cos(theta)
        y = center_point[1] + scaling_factor * bounding_radius_angstroms * np.sin(phi) * np.sin(theta)
        z = center_point[2] + scaling_factor * bounding_radius_angstroms * np.cos(phi)

        points.append(np.array([x, y, z]))

    return points

def write_points_to_PDB(points_array, points_pdb_file):
    """
    Writes a points array into a PDB file as a collection of dummy atoms.

    Inputs:
    - @param[in] points_array (list of numpy.ndarray): the list of points (each point a 3D numpy array)
    - @param[in] points_pdb_file (str): path to the output PDB file to write the points array

    Outputs:
    - @write[out] points_pdb_file : writes the points array to a PDB file saved at this location
    """

    with open(points_pdb_file, "w") as f:
        for i, point in enumerate(points_array):
            x, y, z = point

            # creating a dummy PDB line for the pseudo-point
            pdb_line = (
                f"HETATM {i+1:5d} DU  DUM A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X\n"
            )
            f.write(pdb_line)

        f.write("END\n")

# ********************************** PREPROCESSING WRAPPER FUNCTION ********************************** #

def preprocess_inputs(protein_pdb_path, output_protein_pdb_path, SMILES_string, N_CONFIGURATIONS, system_pH, 
    steric_clash_distance, MBS_scaling_factor, output_BS_shell_path):
    """
    Preprocess both the protein (from its PDB file) and the micromolecule (from its SMILES string) and returns the micromolecule's topology, 
    atomic positions, its corresponding OpenFF object, and writes the computed bounding sphere's shell as a PDB file (for visualization).

    Inputs:
    - @param[in] protein_pdb_path (str) : path to the input protein PDB file
    - @param[in] output_protein_pdb_path (str) : path to a PDB file to save the updated protein structure
    - @param[in] SMILES_string (str) : the SMILES string of the micromolecule
    - @param[in] N_CONFIGURATIONS (int) : the number of protein-micromolecule configurations to generate
    - @param[in] system_pH (float) : pH value for the protein's protonation state (default is biological pH -- 7.4)
    - @param[in] steric_clash_distance (float) : the threshold upper bound distance (in angstroms) that defines steric clashes
    - @param[in] MBS_scaling_factor (float) : controllable factor by which to scale the MBS radius
    - @param[in] output_BS_shell_path (str) : path to a PDB file to save the bounding sphere's outer shell
    
    Outputs:
    - @return (list) : a list of tuples containing the micromolecule's topology (Topology object), atomic positions (numpy.ndarray), and OpenFF object
        for each configuration; this function also writes the bounding sphere's shell as a PDB file
    """
    
    preprocess_protein(protein_pdb_path, output_protein_pdb_path=output_protein_pdb_path, pH=system_pH)

    system_configurations = []
    for _ in range(1, N_CONFIGURATIONS + 1):
        # sampling a computed bounding sphere and instantiating the micromolecule within this sphere
        sampled_micromolecule_position, bs_center, bs_radius = sample_bounding_sphere(output_protein_pdb_path, scaling_factor=MBS_scaling_factor)
        micromolecule_topology, micromolecule_positions, micromolecule_openff = instantiate_micromolecule(SMILES_string, sampled_micromolecule_position)

        while (check_pairwise_steric_clashes(output_protein_pdb_path, micromolecule_positions, steric_clash_distance)):
            # if there exists any steric clashes in the initial configuration between the protein and micromolecule, resample the micromolecule's atomic positions
            sampled_micromolecule_position, _, _ = sample_bounding_sphere(output_protein_pdb_path, scaling_factor=MBS_scaling_factor)
            micromolecule_topology, micromolecule_positions, micromolecule_openff = instantiate_micromolecule(SMILES_string, sampled_micromolecule_position)

        # saving the micromolecule's topology and atomic positions for system setup
        system_configurations.append((micromolecule_topology, micromolecule_positions, micromolecule_openff))

    # visualizing the bounding sphere (for completeness)
    bs_shell_positions = generate_spherical_shell(bs_center, bs_radius, num_points=500)
    write_points_to_PDB(bs_shell_positions, output_BS_shell_path)

    return system_configurations

# **************************************************************************************************** #
