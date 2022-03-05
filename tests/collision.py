'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"

box = BoxBase((100,100))
atoms = Particles(2,2)
atoms.positions = np.array([[10.,90.], [90.,10.]])
atoms.velocities = 5e-2*np.array([[1.,-1.], [-1.,1.]])
worker = NBodyWorker(atoms, box, timestep=0.1)
worker.evolve(800, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.plot(0,len(plotter.history.times)+1)
plotter.show()