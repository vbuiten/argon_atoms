'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
import numpy as np

#savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
savepath = "/net/vdesk/data2/buiten/COP/"

box = BoxBase((100,100))
atoms = Particles(2,2)
atoms.positions = np.array([[30.,50], [60.,51.]])
atoms.velocities = 1.e-3*np.array([[1.,0], [-1.,0.]])
worker = NBodyWorker(atoms, box, timestep=0.0001)
worker.evolve(500, timestep_external=10, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.plot(0,len(plotter.history.times)+1)
plotter.show()