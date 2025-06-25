"""
This file contains utilities relevant for setting up temporal molecular dynamics simulations with the OpenMM framework.

Author: Chetan Chilkunda
Date Created: 16 May 2025
Date Modified: 27 May 2025
"""

# **************************************************************************************************** #

import numpy as np

from openmm.app import PDBFile
from openmm.app import Modeller, ForceField, Simulation
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from openmm import LangevinIntegrator, Platform, XmlSerializer

from openmm.unit import angstrom, nanometer, molar, kelvin, picosecond

# ******************************** OPENMM SIMULATION SETUP FUNCTIONS ******************************** #

def setup_system(protein_pdb_path, micromolecule_topology, micromolecule_positions, micromolecule_openff, output_system_pdb_path, 
    system_pH=7.4, system_ionic_strength=0.15, box_padding=2.5, neutralize_flag=True, positive_ion="Na+", negative_ion="Cl-"):
    """
    Unifies the protein-micromolecule system by (1) combining topologies, (2) creating and merging force fields, (3) setting system pH (by adding hydrogens),
    (4) solvating the system with a specified ionic strength, and (5) writing the system into a PDB file (for visualization). This function returns the
    unified 'modeller' OpenMM object and the OpenMM force field object.

    - @NOTE: the protein and solvent force fields are preset (hard-coded) here for simplicity

    Inputs:
    - @param[in] protein_pdb_path (str) : path to the input protein PDB file
    - @param[in] micromolecule_topology (Topology) : the micromolecule's topology object
    - @param[in] micromolecule_positions (numpy.ndarray) : the micromolecule's atomic positions
    - @param[in] micromolecule_openff (OpenFF) : the micromolecule's OpenFF object
    - @param[in] output_system_pdb_path (str) : path to a PDB file to save the unified system's atomic positions
    - @param[in] system_pH (float) : pH value for the system's protonation state (default is biological pH -- 7.4)
    - @param[in] system_ionic_strength (float) : the ionic strength of the system (default is biological 0.15 M)
    - @param[in] box_padding (float) : the box padding (in angstroms) to add to the system's longest length scale (default is 2.5 angstrom padding)
    - @param[in] neutralize_flag (bool) : whether or not to neutralize the system charge state (default is True)
    - @param[in] positive_ion (str) : the positive ion to use for neutralization (default is Na+)
    - @param[in] negative_ion (str) : the negative ion to use for neutralization (default is Cl-)

    Outputs:
    - @return (tuple) : a tuple containing the unified OpenMM modeller object and the OpenMM force field object
    """

    protein_PDB = PDBFile(protein_pdb_path)

    # combining both the protein and micromolecule into a unified OpenMM system object
    modeller = Modeller(protein_PDB.topology, protein_PDB.positions)
    modeller.add(micromolecule_topology, micromolecule_positions * angstrom)

    # @NOTE: for simplicity, using AMBER ff14SB for the protein's force field and a custom force field for the micromolecule
    # @NOTE: using water 'tip3p' force field for the solvent
    force_field = ForceField("amber/ff14SB.xml", "amber/tip3p_standard.xml")
    smirnoff_object = SMIRNOFFTemplateGenerator(molecules=micromolecule_openff)
    # registering the SMIRNOFF template generator micromolecule force field
    force_field.registerTemplateGenerator(smirnoff_object.generator)

    # adding hydrogens at physiological pH
    modeller.addHydrogens(force_field, pH=system_pH)

    # solvating the system with water in periodic box dimensions with padding along the protein's longest dimension
    # @NOTE: mimicking physiological (blood) conditions with ionic strength of about 0.15 M and Na+ and Cl- as the most abundant ions
    modeller.addSolvent(force_field, model="tip3p", padding=box_padding*angstrom, 
        ionicStrength=system_ionic_strength*molar, neutralize=neutralize_flag, positiveIon=positive_ion, negativeIon=negative_ion)
    
    # adding any missing / extra particles that are required by the force field
    modeller.addExtraParticles(force_field)

    # centering all the atomic coordinates to the origin
    modeller_positions = np.array([pos.value_in_unit(angstrom) for pos in modeller.positions]) # in angstroms
    system_centroid = modeller_positions.mean(axis=0)
    modeller.positions = (modeller_positions - system_centroid) * angstrom

    # writing the complete system into a PDB file (for visualization)
    with open(output_system_pdb_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    return modeller, force_field

def setup_simulation(modeller, force_field, non_bonding_model, non_bonding_cutoff_distance, bond_constraints,
    temperature=310.15, friction_coefficient=1.0, step_size=0.001, hardware_platform="CPU"):
    """
    Aggregates the protein-micromolecule system model and its unified force field into a simulation object and defines (1) the distance cutoff 
    for long-range electrostatic interactions, (2) the dynamics model (Langevin), and (3) the system parameters (i.e temperature, integrator,
    step size, etc.). This simulation object, along with its components, are returned for downstream temporal molecular dynamics simulations.

    Inputs:
    - @param[in] modeller (Modeller) : the OpenMM modeller object for the system
    - @param[in] force_field (ForceField) : the OpenMM force field object for the system
    - @param[in] non_bonding_model (str) : the non-bonding interaction model for the system (options are NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, PME, or LJPME)
    - @param[in] non_bonding_cutoff_distance (float) : the distance cutoff for long-range electrostatic interactions (in nanometers)
    - @param[in] bond_constraints (list) : the bond constraints on the system (options are None, HBonds, AllBonds, or HAngles)
    - @param[in] temperature (float) : the system's temperature (in Kelvin) (default is biological 310.15 K)
    - @param[in] friction_coefficient (float) : friction coefficient that connotes how strongly the system is coupled to the heat bath (in inverse picoseconds with default 1.0 / ps)
    - @param[in] step_size (float) : the integrator step size (in picoseconds) (default is 0.001 ps = 1.0 fs)
    - @param[in] hardware_platform (str) : the hardware platform for computation (default is CPU)

    Outputs:
    - @return (tuple) : a tuple containing the OpenMM simulation object, the OpenMM modeller object, the OpenMM system object, 
        the Langevin integrator object, and the OpenMM platform object
    """

    # modeling the system by merging the topology with the aggregated force field, and defining a distance cutoff for the long-range electrostatic interactions
    # @NOTE: openmm.app.forcefield.ForceField.createSystem() documentation:
    #   http://docs.openmm.org/latest/api-python/generated/openmm.app.forcefield.ForceField.html#openmm.app.forcefield.ForceField.createSystem
    system = force_field.createSystem(modeller.topology, nonbondedMethod=non_bonding_model, nonbondedCutoff=non_bonding_cutoff_distance*nanometer, constraints=bond_constraints)

    # @NOTE: LangevinIntegrator() suited for constant temperature (thermostat) system with moderate damping / resistance to particle motion (and including inertial terms)
    # @NOTE: intertial terms matters at the atomic scale because atoms have mass, experience acceleration, and undergo ballistic and vibrational motions
    integrator = LangevinIntegrator(temperature*kelvin, friction_coefficient/picosecond, step_size*picosecond)

    platform = Platform.getPlatformByName(hardware_platform)

    # creating the simulation object with all its components and specifying the initial velocities for each atom in the system
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature*kelvin) # sampling the Boltzmann distribution to set initial velocities of all the atoms

    simulation.minimizeEnergy() # finding a low-energy (stable) system structure by removing steric clashes / bad geometries

    return simulation, modeller, system, integrator, platform

def checkpoint_simulation(simulation, modeller, system, integrator, 
    output_simulation_checkpoint_path, output_simulation_pdb_path, output_simulation_system_path, output_simulation_integrator_path):
    """
    Checkpoints the OpenMM simulation object to a binary .chk file, writes the OpenMM modeller object into a PDB file, 
    and serializes the OpenMM system and integrator objects into .xml files for downstream usage.

    - @NOTE: the OpenMM platform object cannot be saved via serialization and, thus, it must be recreated upon re-instantiating the checkpoint

    Inputs:
    - @param[in] simulation (Simulation) : the OpenMM simulation object for the system
    - @param[in] modeller (Modeller) : the OpenMM modeller object for the system
    - @param[in] system (System) : the OpenMM system object
    - @param[in] integrator (Integrator) : the OpenMM integrator object for the system
    - @param[in] output_simulation_checkpoint_path (str) : path to a .chk binary file to save the simulation's configuration
    - @param[in] output_simulation_pdb_path (str) : path to a PDB file to save the simulation's structure
    - @param[in] output_simulation_system_path (str) : path to a .xml file to save the simulation's OpenMM system object
    - @param[in] output_simulation_integrator_path (str) : path to a .xml file to save the simulation's OpenMM integrator object

    Outputs:
    - @write[out] output_simulation_checkpoint_path, output_simulation_pdb_path, output_simulation_system_path, output_simulation_integrator_path :
        - (1) writes the simulation to a binary .chk file
        - (2) writes the simulation's topology and positions to a PDB
        - (3) serializes the system object to a .xml file
        - (4) serializes the integrator object to a .xml file
    """
    # @NOTE: .saveCheckpoint() saves a binary file with all information, including random states, etc.
    # @NOTE: .saveCheckpoint() is not portable across hardware and must be used consciously with the same hardware 
    simulation.saveCheckpoint(output_simulation_checkpoint_path)

    # writing the simulation system to a PDB file (for visualization)
    with open(output_simulation_pdb_path, "w") as f:
        PDBFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom), f)

    # saving the system and integrator objects via serialization
    with open(output_simulation_system_path, "w") as f: f.write(XmlSerializer.serialize(system))
    with open(output_simulation_integrator_path, "w") as f: f.write(XmlSerializer.serialize(integrator))

def reinitialize_simulation(simulation_checkpoint_path, simulation_pdb_path, simulation_system_path, simulation_integrator_path, hardware_platform="CPU"):
    """
    Reinitializes the simulation configuration from the (1) .chk binary simulation checkpoint, (2) the PDB file corresponding to the OpenMM modeller
    (and its stored topology and atomic positions), (3) an .xml file corresponding to the OpenMM system object, and (4) an .xml file corresponding 
    to the OpenMM integrator object. Reinitialization recreates the necessary OpenMM objects needed to resume molecular dynamics simulations.
    
    Inputs:
    - @param[in] simulation_checkpoint_path (str) : path to a .chk binary file that contains the simulation's configuration
    - @param[in] simulation_pdb_path (str) : path to a PDB file that contains the simulation's structure
    - @param[in] simulation_system_path (str) : path to a .xml file that contains the simulation's OpenMM system object
    - @param[in] simulation_integrator_path (str) : path to a .xml file that contains the simulation's OpenMM integrator object
    - @param[in] hardware_platform (str) : the hardware platform for computation (default is CPU)

    Outputs:
    - @return (tuple) : a tuple containing the OpenMM simulation object, the OpenMM modeller object, the OpenMM system object, 
        the Langevin integrator object, and the OpenMM platform object
    """

    # using the PDB file to extract the simulation's topology and atomic positions
    simulation_PDB = PDBFile(simulation_pdb_path)
    # reinstantiating the simulation's OpenMM modeller object
    modeller = Modeller(simulation_PDB.topology, simulation_PDB.positions.value_in_unit(angstrom))

    # reloading (deserializing) the system and integrator objects
    with open(simulation_system_path, "r") as f: system = XmlSerializer.deserialize(f.read())
    with open(simulation_integrator_path, "r") as f: integrator = XmlSerializer.deserialize(f.read())

    # redefining the platform hardware
    platform = Platform.getPlatformByName(hardware_platform)

    # reinitializing the complete OpenMM simulation object
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.loadCheckpoint(simulation_checkpoint_path)

    return simulation, modeller, system, integrator, platform

# **************************************************************************************************** #
