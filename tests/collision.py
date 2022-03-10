'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

box = BoxBase((100,100))
atoms = Particles(2,2)
atoms.positions = np.array([[2.,4.9], [18.,5.1]])
atoms.velocities = np.array([[0.09,0], [-0.09,0.]])
worker = NBodyWorker(atoms, box, timestep=0.01)
worker.evolve(10, timestep_external=0.1, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.plot(0,len(plotter.history.times)+1)
plotter.show()

energy_plotter = EnergyPlotter(savepath+"collision-test.hdf")
energy_plotter.plotAll()
energy_plotter.show()