'''Test for experimenting with equilibriation.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from simulation.utils import minimumImageForces
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np
import matplotlib.pyplot as plt

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

n_atoms = 32

box = BoxBase(0.2, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 0.5

forces = minimumImageForces(atoms.positions, box.lengths)
print(atoms.positions)
#print(forces*0.01)

#print ("Initial velocities:", atoms.velocities)
workerVerlet = NBodyWorker(atoms, box, timestep=0.01, minimage=True)
#Efracs = workerVerlet.equilibriate(iterations=10, threshold=0.1, iteration_time=50)
workerVerlet.evolve(20, timestep_external=0.1, savefile=savepath+"verlet-test.hdf")
#print ("Velocities after integrating:", atoms.velocities)

fig, ax = plt.subplots()
#ax.plot([i for i in range(len(Efracs))], Efracs)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\frac{E_{target}}{E_{kin}}$")
fig.show()

plotterVerlet = TrajectoryPlotter(savepath+"verlet-test.hdf")
#plotterVerlet.plot(-10,len(plotterVerlet.history.times)+1)
plotterVerlet.scatter(0,5)
plotterVerlet.fig.suptitle("First 5 steps saved")
plotterVerlet.ax.set_title(r"$dt = 0.1$")
plotterVerlet.show()

plotterVerlet.ax.cla()
plotterVerlet.plot(-10, len(plotterVerlet.history.times)+1)
plotterVerlet.fig.suptitle("Last 10 steps saved")
plotterVerlet.show()

energy_plotterVerlet = EnergyPlotter(savepath+"verlet-test.hdf", dimless=True)
energy_plotterVerlet.plotAll()
energy_plotterVerlet.fig.suptitle("Energy Evolution with Verlet Integration")
energy_plotterVerlet.ax.set_yscale("linear")
energy_plotterVerlet.show()
