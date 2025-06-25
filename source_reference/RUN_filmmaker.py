"""
This file reads in a specified simulation outputs directory and postprocesses all checkpointed PDB files to create a trajectory 'movie' PDB.

Author: Chetan Chilkunda
Date Created: 2 June 2025
Date Modified: 2 June 2025
"""

# **************************************************************************************************** #

import os
import re
import argparse

from UTILS_postprocess import *

# **************************************************************************************************** #

def main():
    """
    Main method reads in input and output directories and creates a trajectory 'movie' from the simulation snapshots saved in the input directory.
    """

    # using the argparse library to efficiently read and manage user inputs
    parser = argparse.ArgumentParser(description="Method for generating a 'movie' PDB file from a directory of simulation PDB snapshot files.", add_help=False)
    parser.add_argument("--help", action="help", help="Showing the following help message.")

    parser.add_argument("--input_dir", type=str, help="Path to the input directory of all simulation PDB snapshot files.")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save all generated files and the trajectory 'movie'.")

    args = parser.parse_args() # parsing the input arguments

    input_dir                       = args.input_dir
    output_dir                      = args.output_dir

    # defining a regular expression pattern for identifying simulation snapshot (checkpoint) files
    # @NOTE: regular expression pattern matches the naming convention used
    regex_pattern = re.compile(r"checkpoint.*?(\d+).*?\.pdb$")
    # identifying all simulation snapshot (checkpoint) files for movie generation
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if regex_pattern.search(f)]

    trajectory_states = []
    for input_file in input_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        output_file = output_file.replace(".pdb", "_bare.pdb")

        postprocess_PDB_file(input_file, output_file, exclusions_list=["HOH", "NA", "CL"])
        trajectory_states.append(output_file)

    print(f"\n\t--> post-processed the generated data from {input_dir}...\n")

    simulation_trajectory_path = os.path.join(output_dir, f"trajectory.pdb")
    create_trajectory_from_PDBs(trajectory_states, simulation_trajectory_path)

    print(f"\t--> created the simulation trajectory... DONE.\n")

# **************************************************************************************************** #

if __name__ == "__main__": main()

# **************************************************************************************************** #
