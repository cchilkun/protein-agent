"""
This file reads in several user-defined simulation parameters and performs the following actions:
    - (1) reinitializes the simulation from a previously saved checkpoint
    - (2) if thermal equilibration is required, samples the Boltzmann distribution to specify all initial velocities
    - (3) creates an OpenMM StateDataReporter to log simulation energetics and properties
    - (4) runs the temporal MD loop and writes checkpoint files to save progress

Author: Chetan Chilkunda
Date Created: 16 May 2025
Date Modified: 2 June 2025

@NOTE: make sure if resuming a run then to check the StateDataReporter file name (to avoid overwriting previously generated data)

Usage: run 'python ./RUN_molecular_dynamics.py --help' for complete documentation (assuming cd-ed into the directory containing these files)
"""

# **************************************************************************************************** #

import os
import argparse

import numpy as np

from openmm.app import StateDataReporter # for logging metrics

from openmm.unit import kelvin

from UTILS_simulate import *

# **************************************************************************************************** #

def main():
    """
    Main method reads in all user inputs and runs temporal molecular dynamics (with thermal equilibration if needed).
    """

    # using the argparse library to efficiently read and manage user inputs
    parser = argparse.ArgumentParser(description="Method for running temporal molecular dynamics on the given protein-micromolecule simulation system.", add_help=False)
    parser.add_argument("--help", action="help", help="Showing the following help message.")

    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save all generated files.")

    parser.add_argument("--sim_chkpt_binary_path", type=str, help="Path to a simulation's stored checkpoint binary .chk file.")
    parser.add_argument("--sim_chkpt_pdb_path", type=str, help="Path to a simulation's stored checkpoint PDB file.")
    parser.add_argument("--sim_chkpt_system_xml_path", type=str, help="Path to a simulation's serialized OpenMM system object XML file.")
    parser.add_argument("--sim_chkpt_integrator_xml_path", type=str, help="Path to a simulation's serialized OpenMM integrator object XML file.")
    parser.add_argument("--hardware_platform", type=str, default="CPU", help="Hardware platform for computation.")

    parser.add_argument("--step_start_index", type=int, default=0, help="Starting index (characteristic time step number) for the simulation.")
    parser.add_argument("--num_steps", type=int, help="Number of characteristic time steps to simulate (in femtoseconds).")
    parser.add_argument("--logging_interval", type=int, help="Interval of number of characteristic time steps (in femtoseconds) to skip before logging metrics.")
    # default 'temperature_flag' set to -1.0 to specify if the simulation system is in thermal equilibration or not (and casing on positive versus negative)
    parser.add_argument("--temperature_flag", type=float, default=-1.0, help="Temperature of the simulation system -- only specify if starting a new MD simulation.")

    args = parser.parse_args() # parsing the input arguments

    output_dir                      = args.output_dir
    sim_chkpt_binary_path           = args.sim_chkpt_binary_path
    sim_chkpt_pdb_path              = args.sim_chkpt_pdb_path
    sim_chkpt_system_xml_path       = args.sim_chkpt_system_xml_path
    sim_chkpt_integrator_xml_path   = args.sim_chkpt_integrator_xml_path
    hardware_platform               = args.hardware_platform
    step_start_index                = args.step_start_index
    num_steps                       = args.num_steps
    logging_interval                = args.logging_interval
    temperature_flag                = args.temperature_flag

    simulation, modeller, system, integrator, platform = reinitialize_simulation(
        sim_chkpt_binary_path, sim_chkpt_pdb_path, sim_chkpt_system_xml_path, sim_chkpt_integrator_xml_path, hardware_platform)

    # @NOTE: change the name of this logging .csv file if resuming a simulation run to not override previous data!
    if (temperature_flag >= 0): # if simulation system is not currently in thermal equilibrium, need to sample Boltzmann distribution to set initial velocities
        simulation.context.setVelocitiesToTemperature(temperature_flag*kelvin)
        log_file_csv_path = os.path.join(output_dir, f"temperature_trajectory_log.csv")
    else:
        log_file_csv_path = os.path.join(output_dir, f"simulation_trajectory_log.csv")

    # logging energetics metrics and simulation system properties via OpenMM's StateDataReporter
    simulation.reporters.append(
        StateDataReporter(
            log_file_csv_path,
            reportInterval      = logging_interval,     # interval (in characteristic femtosecond time steps) at which to write frames
            step                = True,                 # current step index
            time                = True,                 # current time
            potentialEnergy     = True,
            kineticEnergy       = True,
            totalEnergy         = True,
            temperature         = True,                 # instantaneous temperature
            volume              = True,                 # periodic box volume
            density             = True,                 # system density
            progress            = True,                 # current progress (percentage completion)
            speed               = True,                 # estimate of the simulation speed (ns/day)
            elapsedTime         = True,                 # elapsed time since start of simulation (seconds)
            separator           = ",",
            totalSteps          = step_start_index+num_steps,
        )
    )

    # manually running the temporal molecular dynamics loop (for ease in logging atomic positions)
    for step in range(step_start_index, step_start_index+num_steps, logging_interval):
        chkpt_binary_path           = os.path.join(output_dir, f"checkpoint_{step}.chk")
        chkpt_pdb_path              = os.path.join(output_dir, f"checkpoint_{step}.pdb")
        chkpt_system_xml_path       = os.path.join(output_dir, f"checkpoint_system.xml")
        chkpt_integrator_xml_path   = os.path.join(output_dir, f"checkpoint_integrator.xml")

        # checkpointing the simulation system
        checkpoint_simulation(simulation, modeller, system, integrator, 
            chkpt_binary_path, chkpt_pdb_path, chkpt_system_xml_path, chkpt_integrator_xml_path)

        print(f"\t--> logging step #{step}...\t\t\t {np.round(100 * (step / (step_start_index+num_steps)), 3)}% completed...")

        simulation.step(logging_interval)

    chkpt_binary_path           = os.path.join(output_dir, f"checkpoint_{step+logging_interval}.chk")
    chkpt_pdb_path              = os.path.join(output_dir, f"checkpoint_{step+logging_interval}.pdb")
    chkpt_system_xml_path       = os.path.join(output_dir, f"checkpoint_system.xml")
    chkpt_integrator_xml_path   = os.path.join(output_dir, f"checkpoint_integrator.xml")

    # checkpointing the simulation system
    checkpoint_simulation(simulation, modeller, system, integrator, 
        chkpt_binary_path, chkpt_pdb_path, chkpt_system_xml_path, chkpt_integrator_xml_path)

    print(f"\t--> logging step #{step+logging_interval}...\t\t\t {np.round(100 * ((step+logging_interval) / (step_start_index+num_steps)), 3)}% completed...")

    print(f"\n\t--> finished running temporal molecular dynamics... DONE.\n")

# **************************************************************************************************** #

if __name__ == "__main__": main()

# **************************************************************************************************** #
