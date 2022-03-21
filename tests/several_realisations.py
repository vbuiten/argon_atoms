'''Test file for experimenting with running several realisations of the same initial conditions.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import Simulator
from analysis.correlation import DistanceHistogram, CorrelationFunction
from analysis.pressure import VirialPressure

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

n_atoms = 108
iterations = 20
temperature = 3.0
density = 0.3

box = BoxBase(density, n_atoms, 3)

particle_sets = []

for i in range(iterations):
    atoms = Particles(n_atoms,3)
    atoms.positions = box.edges
    atoms.temperature = temperature

    sim = Simulator(atoms, box, timestep=0.001)
    sim.equilibrate(iteration_time=5, threshold=0.01)
    sim.evolve(10, timestep_external=1., savefile=savepath+"liquid-iteration"+str(i+1)+".hdf")

    particle_sets.append(atoms)

    print ("Iteration "+str(i+1)+" finished.")

distance_hist = DistanceHistogram(particle_sets, box.lengths, bins=100)
distance_hist.plotIterationAveraged()
distance_hist.show()

corr_func = CorrelationFunction(particle_sets, box.lengths, bins=100)
corr_func.plot()
corr_func.show()

pressure = VirialPressure(particle_sets, box.lengths)
pressure.plot()
pressure.show()