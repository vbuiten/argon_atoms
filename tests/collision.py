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
atoms.positions = np.array([[40.,50], [60.,50.1]])
atoms.velocities = 1.e-2*np.array([[1.,0], [-1.,0.]])
worker = NBodyWorker(atoms, box, timestep=0.01)
worker.evolve(1000, timestep_external=10, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.plot(0,len(plotter.history.times)+1)
plotter.show()