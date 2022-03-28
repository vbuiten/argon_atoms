# Molecular Dynamics Simlations of Argon

This repository contains the code used for Project 1 of the course Computational Physics 2022 at Leiden University. It
is written by Victorine Buiten.

Last updated: March 28 2022

##Repository Contents

The repository is divided into folders for different functionalities of the code. These are the following:
* _analysis_ contains several files of code for analysing simulation results, e.g. for plotting the pair correlation 
function.
* _data_ contains utility functions for handling file paths and I/O operations.
* _framework_ contains code for the base Particles and BoxBase classes, to which the simulations apply.
* _phase_diagram_simulations_ contains code for running simulations used for creating a phase diagram.
* _simulation_ contains code for setting up and running the simulations.
* _tests_ contains code for running various tests of the code, such as whether energy is conserved and how the simulation
behaves for a small number of particles.

##Dependencies

The simulation code uses only the NumPy and Numba packages. For the analysis, SciPy and Matplotlib are also required.

##How to Run the Code

Any simulation requires the construction of a Particles instance and a BoxBase instance, both of which can be found in /framework.
The Particles instance must also be given a set of positions and velocities, OR an array of box edges and a dimensionless temperature. For example:

===

n_atoms = 108

box = BoxBase(0.8, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 1.0

===

The simulation can be run by using a Simulator object:

===

workerVerlet = Simulator(atoms, box, timestep=0.001, minimage=True, method="Verlet")

Efracs = workerVerlet.equilibrate(iterations=20, threshold=0.01, iteration_time=10)

workerVerlet.evolve(200, timestep_external=0.1, savefile=savepath+"verlet-test.hdf5")

===

Next, we plot the progress in the equilibration steps and the evolution of the energy in the code throughout the simulation:

===

equilibration_plotter = EquilibrationPlotter(Efracs)

equilibration_plotter.plot()

equilibration_plotter.show()

energy_plotter = EnergyPlotter(savepath+"verlet-test.hdf5")

energy_plotter.plotAll()

energy_plotter.show()

===

For examples of how to run and analyse a number of simulations (e.g. construct a pressure histogram), see 
_repeated_simulations_ and _phase_diagram_simulations_.
