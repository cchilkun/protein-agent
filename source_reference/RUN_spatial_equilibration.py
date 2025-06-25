"""
This file reads in several user-defined simulation parameters and performs the following actions:
    - (1) instantiates a protein-micromolecule molecular dynamics simulation
    - (2) spatially equilibrates the system w.r.t all atomic coordinates
    - (3) checkpoints the simulation for downstream temporal simulations

Author: Chetan Chilkunda
Date Created: 16 May 2025
Date Modified: 2 June 2025

Usage: run 'python ./RUN_spatial_equilibration.py --help' for complete documentation (assuming cd-ed into the directory containing these files)
"""

# **************************************************************************************************** #

import os
import argparse

from openmm.app import PME, HBonds # importing the non-bonding interaction model and bond constraints model for the system

from UTILS_preprocess import *
from UTILS_simulate import *

# **************************************************************************************************** #

def main():
    """
    Main method reads in all user inputs and performs the necessary steps for spatial equilibration on the protein-micromolecule simulation system.
    """

    print("\tRunning main() to configure and spatially equilibrate the protein-micromolecule simulation system...\n")

    # using the argparse library to efficiently read and manage user inputs
    parser = argparse.ArgumentParser(description="Method for configuring and spatially equilibrating the given " \
        "protein-micromolecule simulation system for downstream temporal molecular dynamics.", add_help=False)
    parser.add_argument("--help", action="help", help="Showing the following help message.")

    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save all generated files.")
    parser.add_argument("--protein_name", type=str, help="String identifier for the protein (i.e. HSA or Albumin).")
    parser.add_argument("--micromolecule_name", type=str, help="String identifier for the micromolecule (i.e. Ibuprofen or Warfarin).")

    # input flags for the protein PDB file, the micromolecule SMILES string, and integer argument for the number of initial configurations to generate
    parser.add_argument("--input_protein_PDB_path", type=str, help="Path to the input protein PDB file.")
    parser.add_argument("--input_SMILES_string", type=str, help="The SMILES string of the micromolecule.")
    parser.add_argument("--N_CONFIGURATIONS", type=int, default=1, help="The number of protein-micromolecule initial configurations to generate.")
    parser.add_argument("--start_config_index", type=int, default=0, help="The starting index in naming the generated protein-micromolecule initial configurations.")

    # input flags for the simulation's physical parameters
    parser.add_argument("--system_pH", type=float, default=7.4, help="pH value for the system's protonation state.")
    parser.add_argument("--system_ionic_strength", type=float, default=0.15, help="Ionic strength of the system (in M).")
    parser.add_argument("--neutralize_flag", type=bool, default=True, help="Whether or not to neutralize (zero out) the system's net charge with solvent ions.")
    parser.add_argument("--temperature", type=float, default=310.15, help="Temperature of the system (in K).")

    # input flags for the simulation's modelling assumptions
    parser.add_argument("--friction_coefficient", type=float, default=1.0, help="Friction coefficient (in inverse picoseconds).")
    parser.add_argument("--steric_clash_distance", type=float, default=1.0, help="Steric clash distance threshold (in angstroms).")
    parser.add_argument("--box_padding", type=float, default=2.5, help="Padding (in angstroms) for surrounding the system with a padded periodic square box.")
    parser.add_argument("--non_bonding_cutoff_distance", type=float, default=1.0, help="Distance cutoff that defines the long-range electrostatic interactions (in nanometers).")
    parser.add_argument("--mbs_scaling_factor", type=float, default=1.0, help="Parameter by which to scale the computed minimum-bounding-sphere's radius.")

    # input flags for the simulation's parameters
    parser.add_argument("--step_size", type=float, default=0.001, help="Integrator step size (in picoseconds).")
    parser.add_argument("--hardware_platform", type=str, default="CPU", help="Hardware platform for computation.")

    args = parser.parse_args() # parsing the input arguments

    output_dir                      = args.output_dir
    protein_name                    = args.protein_name
    micromolecule_name              = args.micromolecule_name
    input_protein_PDB_path          = args.input_protein_PDB_path
    input_SMILES_string             = args.input_SMILES_string
    N_CONFIGURATIONS                = args.N_CONFIGURATIONS
    start_config_index              = args.start_config_index
    system_pH                       = args.system_pH
    system_ionic_strength           = args.system_ionic_strength
    neutralize_flag                 = args.neutralize_flag
    temperature                     = args.temperature
    friction_coefficient            = args.friction_coefficient
    steric_clash_distance           = args.steric_clash_distance
    mbs_scaling_factor              = args.mbs_scaling_factor
    box_padding                     = args.box_padding
    non_bonding_cutoff_distance     = args.non_bonding_cutoff_distance
    step_size                       = args.step_size
    hardware_platform               = args.hardware_platform

    # assigning the non-bonding interaction model (Particle Mesh Ewald) and bond constraints (fixed Hydrogen bonds) for simulation performance speedup
    non_bonding_model               = PME
    bond_constraints                = HBonds

    # writing names for the generated protein-specific output files (excluding the micromolecule and its associated configurations)
    cleaned_protein_pdb_path        = os.path.join(output_dir, f"{protein_name}_CLEANED.pdb")
    bs_shell_pdb_path               = os.path.join(output_dir, f"{protein_name}_MBS_SHELL.pdb")

    ''' configuring the protein-micromolecule simulation system and performing spatial equilibration '''
    micromolecule_configurations = \
        preprocess_inputs(input_protein_PDB_path, cleaned_protein_pdb_path, input_SMILES_string, 
            N_CONFIGURATIONS, system_pH, steric_clash_distance, mbs_scaling_factor, bs_shell_pdb_path)
    
    print(f"\t--> preprocessed the protein's PDB file...\n")

    for i, (micromolecule_topology, micromolecule_positions, micromolecule_openff) in enumerate(micromolecule_configurations):
        print(f"\t\tmicromolecule configuration #{start_config_index+i+1}:")

        # creating a specific sub-directory for each generated protein-micromolecule initial configuration
        config_subdir = os.path.join(output_dir, f"{protein_name}_{micromolecule_name}_CONFIG_{start_config_index+i+1}")
        os.mkdir(config_subdir)

        # writing the micromolecule's sampled initial configuration to a PDB file for visualization
        micromolecule_viz_pdb_path = os.path.join(config_subdir, f"INITIAL_PLACEMENT.pdb")
        write_micromolecule_to_PDB(micromolecule_topology, micromolecule_positions, micromolecule_viz_pdb_path)

        print(f"\t--> wrote the micromolecule's sampled initial configuration to a PDB file...")

        # setting up the simulation system and writing the spatial configuration to a PDB file for visualization
        simulation_config_viz_pdb_path = os.path.join(config_subdir, f"INITIAL_CONFIGURATION.pdb")
        modeller, force_field = setup_system(cleaned_protein_pdb_path, micromolecule_topology, micromolecule_positions, micromolecule_openff, 
            simulation_config_viz_pdb_path, system_pH, system_ionic_strength, box_padding, neutralize_flag, positive_ion="Na+", negative_ion="Cl-")

        print(f"\t--> configured the simulation system with a unified protein-micromolecule object and its force field...")

        # setting up the simulation and spatially equilibrating all atomic positions
        simulation, modeller, system, integrator, _ = setup_simulation(modeller, force_field, non_bonding_model, non_bonding_cutoff_distance, 
            bond_constraints, temperature, friction_coefficient, step_size, hardware_platform)
        
        print(f"\t--> spatially equilibrated the simulation system... the eagle has landed...")

        # checkpointing the spatially equilibrated simulation system
        chkpt_binary_path           = os.path.join(config_subdir, f"spatial_equilibration.chk")
        chkpt_pdb_path              = os.path.join(config_subdir, f"spatial_equilibration.pdb")
        chkpt_system_xml_path       = os.path.join(config_subdir, f"spatial_equilibration_system.xml")
        chkpt_integrator_xml_path   = os.path.join(config_subdir, f"spatial_equilibration_integrator.xml")
        checkpoint_simulation(simulation, modeller, system, integrator,
            chkpt_binary_path, chkpt_pdb_path, chkpt_system_xml_path, chkpt_integrator_xml_path)

        print(f"\t--> check-pointed all system components... DONE.\n")

# **************************************************************************************************** #

if __name__ == "__main__": main()

# **************************************************************************************************** #
