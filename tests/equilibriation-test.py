'''Test for experimenting with equilibriation.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np
import matplotlib.pyplot as plt

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

n_atoms = 10

box = BoxBase(0.3, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 1.0
#print ("Initial velocities:", atoms.velocities)
workerVerlet = NBodyWorker(atoms, box, timestep=0.01, minimage=True)
Efracs = workerVerlet.equilibriate(iterations=5, threshold=0.5)
workerVerlet.evolve(50, timestep_external=0.1, savefile=savepath+"verlet-test.hdf")
#print ("Velocities after integrating:", atoms.velocities)

fig, ax = plt.subplots()
ax.plot([i for i in range(len(Efracs))], Efracs)
ax.set_xlabel("Iteration")
ax.set_ylabel(r"$\frac{E_{target}}{E_{kin}}$")
fig.show()

plotterVerlet = TrajectoryPlotter(savepath+"verlet-test.hdf")
plotterVerlet.plot(-3,len(plotterVerlet.history.times)+1)
plotterVerlet.fig.suptitle("Verlet integration")
plotterVerlet.ax.set_title(r"$dt = 0.01$")
plotterVerlet.show()

energy_plotterVerlet = EnergyPlotter(savepath+"verlet-test.hdf", dimless=True)
energy_plotterVerlet.plotAll()
energy_plotterVerlet.fig.suptitle("Energy Evolution with Verlet Integration")
energy_plotterVerlet.show()