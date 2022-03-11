'''Test for experimenting with equilibriation.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

n_atoms = 10

box = BoxBase(0.3, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 3.0
print ("Initial velocities:", atoms.velocities)
workerVerlet = NBodyWorker(atoms, box, timestep=0.01)
workerVerlet.equilibriate()
workerVerlet.evolve(50, timestep_external=1., savefile=savepath+"verlet-test.hdf")
print ("Velocities after integrating:", atoms.velocities)

plotterVerlet = TrajectoryPlotter(savepath+"verlet-test.hdf")
plotterVerlet.plot(-3,len(plotterVerlet.history.times)+1)
plotterVerlet.fig.suptitle("Verlet integration")
plotterVerlet.ax.set_title(r"$dt = 0.01$")
plotterVerlet.show()

energy_plotterVerlet = EnergyPlotter(savepath+"verlet-test.hdf")
energy_plotterVerlet.plotAll()
energy_plotterVerlet.fig.suptitle("Energy Evolution with Verlet Integration")
energy_plotterVerlet.show()