# argon_atoms

Any simulation requires the construction of a Particles instance and a BoxBase instance, both of which can be found in /framework.
The Particles instance must also be given a set of positions and velocities, OR an array of box edges and a dimensionless temperature. For example:

===
n_atoms = 108

box = BoxBase(0.8, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 1.0

===

The simulation can be run by setting up a Simulator object:

===
workerVerlet = Simulator(atoms, box, timestep=0.001, minimage=True, method="Verlet")
Efracs = workerVerlet.equilibrate(iterations=20, threshold=0.01, iteration_time=10)
workerVerlet.evolve(200, timestep_external=0.1, savefile=savepath+"verlet-test.hdf")

===
