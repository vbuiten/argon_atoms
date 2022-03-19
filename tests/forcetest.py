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

n_atoms = 2

box = BoxBase(0.1, n_atoms, 3)
atoms = Particles(n_atoms,3)
atoms.positions = box.edges
atoms.temperature = 1.0

atoms.positions = np.array([[1.0, 0.2, 0.01],
                            [1.0, 1.4, 0.01]])
atoms.velocities = np.array([[0.01, 0.6, 0.0],
                             [-0.01, -0.6, 0.0]])
forces = minimumImageForces(atoms.positions, box.lengths)
print(forces*0.001)

#print ("Initial velocities:", atoms.velocities)
workerVerlet = NBodyWorker(atoms, box, timestep=0.001, minimage=True)
workerVerlet.evolve(3, timestep_external=0.01, savefile=savepath+"verlet-test.hdf")
#print ("Velocities after integrating:", atoms.velocities)

plotterVerlet = TrajectoryPlotter(savepath+"verlet-test.hdf")
plotterVerlet.plot(0,len(plotterVerlet.history.times)+1)
plotterVerlet.fig.suptitle("Verlet integration")
plotterVerlet.ax.set_title(r"$dt = 0.01$")
plotterVerlet.show()

energy_plotterVerlet = EnergyPlotter(savepath+"verlet-test.hdf", dimless=True)
energy_plotterVerlet.plotKinetic()
energy_plotterVerlet.plotPotential()
energy_plotterVerlet.fig.suptitle("Energy Evolution with Verlet Integration")
energy_plotterVerlet.show()
