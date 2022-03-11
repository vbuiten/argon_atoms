'''Test set-up of a near-hit of two particles in 2D.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import NBodyWorker
from analysis.visualisation import TrajectoryPlotter
from analysis.energies import EnergyPlotter
import numpy as np

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"
#savepath = "/net/vdesk/data2/buiten/COP/"

box = BoxBase((10,10))
atoms = Particles(100,2)
atoms.positions = box.edges
atoms.velocities = 1.
worker = NBodyWorker(atoms, box, timestep=0.01)
worker.evolve(100, timestep_external=1., savefile=savepath+"100-atoms-test.hdf")

plotter = TrajectoryPlotter(savepath+"100-atoms-test.hdf")
plotter.plot(-3,len(plotter.history.times)+1)
plotter.show()

energy_plotter = EnergyPlotter(savepath+"100-atoms-test.hdf")
energy_plotter.plotAll()
energy_plotter.show()