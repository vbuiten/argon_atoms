'''Test file for experimenting with running several realisations of the same initial conditions.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import Simulator
from analysis.correlation import DistanceHistogram, CorrelationFunction
from analysis.pressure import VirialPressure
from analysis.utils import PlotPreferences, SimulationIterations
import os

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"

if not os.access(savepath, os.F_OK):
    savepath = "/net/vdesk/data2/buiten/COP/"

    foldername = "/dens0.3-temp3.0/"
    folderpath = savepath+foldername

else:
    foldername = "\\dens0.3-temp3.0\\"
    folderpath = savepath+foldername

if not os.access(folderpath, os.F_OK):
    os.mkdir(folderpath)

plotprefs = PlotPreferences(usetex=True)

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
    sim.evolve(50, timestep_external=1., savefile=folderpath+"iteration"+str(i+1)+".hdf5")

    print ("Iteration "+str(i+1)+" finished.")

savefilestart = folderpath + "20runs-"

sim_runs = SimulationIterations(folderpath)

distance_hist = DistanceHistogram(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
distance_hist.plotIterationAveraged()
distance_hist.save(savefilestart+"disthist.png")
distance_hist.show()

corr_func = CorrelationFunction(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
corr_func.plot()
corr_func.save(savefilestart+"corrfunc.png")
corr_func.show()

pressure = VirialPressure(sim_runs.final_particles, sim_runs.box.lengths, plotprefs=plotprefs)
pressure.plot()
pressure.save(savefilestart+"pressure.png")
pressure.show()