'''Test for comparing Euler and Verlet integration.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

box = BoxBase(0.3, 5, 3)
atoms = Particles(5,3)
atoms.positions = box.edges
atoms.temperature = 100
print ("Initial velocities:", atoms.velocities)
workerVerlet = NBodyWorker(atoms, box, timestep=0.01)
workerVerlet.evolve(50, timestep_external=1., savefile=savepath+"verlet-test.hdf")
print ("Velocities after integrating:", atoms.velocities)

workerEuler = NBodyWorker(atoms, box, timestep=0.01)
workerEuler.evolve(50, timestep_external=1., savefile=savepath+"euler-test.hdf",
                   method="Euler")


plotterVerlet = TrajectoryPlotter(savepath+"verlet-test.hdf")
plotterVerlet.plot(-3,len(plotterVerlet.history.times)+1)
plotterVerlet.fig.suptitle("Verlet integration")
plotterVerlet.ax.set_title(r"$dt = 0.01$")
plotterVerlet.show()

plotterEuler = TrajectoryPlotter(savepath+"euler-test.hdf")
plotterEuler.plot(-3,len(plotterEuler.history.times)+1)
plotterEuler.fig.suptitle("Euler integration")
plotterEuler.ax.set_title(r"$dt = 0.01$")
plotterEuler.show()

energy_plotterVerlet = EnergyPlotter(savepath+"verlet-test.hdf")
energy_plotterVerlet.plotAll()
energy_plotterVerlet.fig.suptitle("Energy Evolution with Verlet Integration")
energy_plotterVerlet.show()

energy_plotterEuler = EnergyPlotter(savepath+"euler-test.hdf")
energy_plotterEuler.plotAll()
energy_plotterEuler.fig.suptitle("Energy Evolution with Euler Integration")
energy_plotterEuler.show()