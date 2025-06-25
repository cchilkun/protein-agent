"""
This file transforms the OpenMM PDB simulation checkpoints saved in a specified directory for downstream RL.

Author: Chetan Chilkunda
Date Created: 3 June 2025
Date Modified: 9 June 2025
"""

# **************************************************************************************************** #

import os
import argparse

import re

from UTILS_postprocess import *

# **************************************************************************************************** #

def main():
    """ [TODO] """

    parser = argparse.ArgumentParser(description="Method for postprocessing a given protein-micromolecule simulation run for downstream RL training.", add_help=False)
    parser.add_argument("--help", action="help", help="Showing the following help message.")
    
    parser.add_argument("--simulation_dir", type=str, help="Path to a simulation directory containing all simulation checkpoint files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save the generated trajectory.")
    parser.add_argument("--trajectory_name", type=str, help="String identifier by which to save the generated trajectory.")
    parser.add_argument("--checkpoint_start_index", type=int, help="Starting index for reading the checkpoint PBD files.")
    parser.add_argument("--checkpoint_end_index", type=int, help="Ending index for reading the checkpoint PBD files.")

    args = parser.parse_args() # parsing the input arguments

    simulation_dir                  = args.simulation_dir
    output_dir                      = args.output_dir
    trajectory_name                 = args.trajectory_name
    chkpt_start_index               = args.checkpoint_start_index
    chkpt_end_index                 = args.checkpoint_end_index

    regex_pattern = re.compile(r"checkpoint_(\d+)\.pdb$") # using regular expression to filter over the generated simulation files
    simulation_files = filter_directory_by_REGEX(simulation_dir, regex_pattern, chkpt_start_index, chkpt_end_index)
    simulation_files.sort()

    protein_includes_list = ["ARG", "HIS", "LYS", "ASP", "GLU", "SER", "THR", "ASN", "GLN", 
        "GLY", "PRO", "CYS", "SEC", "ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TYR", "TRP"]
    micromolecule_includes_list = ["UNK"]
    includes_list = protein_includes_list + micromolecule_includes_list

    for i, f in tqdm(enumerate(simulation_files), ascii=True, desc="postprocessing checkpoint files"):

        curr_checkpoint_file = PDBFile(simulation_files[i])
        curr_checkpoint_topology = curr_checkpoint_file.getTopology()
        curr_checkpoint_positions = np.array([pos.value_in_unit(angstrom) for pos in curr_checkpoint_file.positions])

        curr_protein_topology, curr_protein_positions = postprocess_PDB_checkpoint(curr_checkpoint_topology, curr_checkpoint_positions, includes_list)

        break



# **************************************************************************************************** #

if __name__ == "__main__": main()

# **************************************************************************************************** #
