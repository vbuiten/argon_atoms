'''File for running 100 realisations of the liquid conditions from lecture 5.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import Simulator
from analysis.correlation import DistanceHistogram, CorrelationFunction
from analysis.pressure import VirialPressure
from analysis.utils import PlotPreferences, SimulationIterations
from data.utils import folderPath

plotprefs = PlotPreferences(usetex=True)

n_atoms = 108
iterations = 100
temperature = 1.0
density = 0.8

box = BoxBase(density, n_atoms, 3)

foldername = "/dens"+str(density)+"-temp"+str(temperature)+"/"
folderpath = folderPath(foldername)

for i in range(iterations):
    atoms = Particles(n_atoms,3)
    atoms.positions = box.edges
    atoms.temperature = temperature

    sim = Simulator(atoms, box, timestep=0.001)
    sim.equilibrate(iteration_time=5, threshold=0.01)
    sim.evolve(100, timestep_external=1., savefile=folderpath+"iteration"+str(i+1)+".hdf5")

    print ("Iteration "+str(i+1)+" finished.")


sim_runs = SimulationIterations(folderpath)

distance_hist = DistanceHistogram(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
distance_hist.plotIterationAveraged()
distance_hist.save(folderpath+"disthist.png")
distance_hist.show()

corr_func = CorrelationFunction(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
corr_func.plot()
corr_func.save(folderpath+"corrfunc.png")
corr_func.show()

pressure = VirialPressure(sim_runs.final_particles, sim_runs.box.lengths, plotprefs=plotprefs)
pressure.plot()
pressure.save(folderpath+"pressure.png")
pressure.show()