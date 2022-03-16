'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

#savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
savepath = "/net/vdesk/data2/buiten/COP/"

dt = 0.00001
speed = 1e-5

box = BoxBase(1e-4, 2, 2)
atoms = Particles(2,2)
atoms.positions = np.array([[2.,70.0], [138.,71.]])
atoms.velocities = np.array([[speed,0.], [-speed,0.]])
worker = NBodyWorker(atoms, box, timestep=dt)
worker.evolve(0.01, timestep_external=dt, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.scatter(0,len(plotter.history.times)+1)
plotter.show()

energy_plotter = EnergyPlotter(savepath+"collision-test.hdf")
energy_plotter.plotAll()
energy_plotter.show()