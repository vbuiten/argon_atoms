'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

#savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
savepath = "/net/vdesk/data2/buiten/COP/"

box = BoxBase((20,20))
atoms = Particles(4,2)
atoms.positions = np.array([[6.,4.9], [14.,5.1], [10.,10.], [10.,2.]])
atoms.velocities = np.array([[0.09,0], [-0.09,0.], [0.,-0.09], [0.,0.09]])
worker = NBodyWorker(atoms, box, timestep=0.01)
worker.evolve(100, timestep_external=1, savefile=savepath+"collision-test.hdf")

plotter = TrajectoryPlotter(savepath+"collision-test.hdf")
plotter.plot(0,len(plotter.history.times)+1)
plotter.show()

energy_plotter = EnergyPlotter(savepath+"collision-test.hdf")
energy_plotter.plotAll()
energy_plotter.show()