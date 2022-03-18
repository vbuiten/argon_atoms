'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

dt = 0.001
speed = 1
n_atoms = 6

box = BoxBase(1e-2, n_atoms, 2)
atoms = Particles(n_atoms,2)
atoms.positions = np.array([[2.,5.0], [10.,5.1], [6.,8.], [6.1,2.], [15.,10.], [13.,10.1]])
atoms.velocities = np.array([[speed,0.], [-speed,0.], [0.,-speed], [0.,speed], [-speed,0.], [speed,0.]])
worker = NBodyWorker(atoms, box, timestep=dt)
worker.evolve(25, timestep_external=0.1, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.scatter(0,len(plotter.history.times)+1)
plotter.show()

energy_plotter = EnergyPlotter(savepath+"collision-test.hdf")
energy_plotter.plotAll()
energy_plotter.ax.set_yscale("linear")
energy_plotter.show()