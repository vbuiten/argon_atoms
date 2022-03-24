'''File for running simulations for a range of initial density and temperature.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import Simulator
from analysis.pressure import PhaseDiagram
from analysis.utils import PlotPreferences, SimulationIterations
from data.utils import folderPath
import numpy as np

plotprefs = PlotPreferences(usetex=True)

n_atoms = 108
sqrt_iterations = 7
iterations = sqrt_iterations**2

'''
temperature = np.linspace(0.3, 3.0, sqrt_iterations)
density = np.linspace(0.3, 1.2, sqrt_iterations)

temperature = np.array([0.1, 3.5])
density = np.array([0.1, 1.5])

temperature = np.array([1.0, 2.0])
density = np.array([0.5, 1.0])
'''

temperature = np.array([1.5, 1.8, 2.3])
density = np.array([0.4, 0.5])

foldername = "/49points-gas-to-solid/"
folderpath = folderPath(foldername)

for i in range(len(temperature)):
    for j in range(len(density)):

        box = BoxBase(density[j], n_atoms, 3)

        atoms = Particles(n_atoms,3)
        atoms.positions = box.edges
        atoms.temperature = temperature[i]

        sim = Simulator(atoms, box, timestep=0.001)
        sim.equilibrate(iteration_time=5, threshold=0.01)
        sim.evolve(100, timestep_external=1.,
                   savefile=folderpath+"dens"+str(np.around(density[j],3))+"-temp"+str(np.around(temperature[i],3))+".hdf5")

        print ("Iteration "+str(i+1)+" finished.")

sim_runs = SimulationIterations(folderpath, samebox=False)
phase_diagram = PhaseDiagram(sim_runs.final_particles, sim_runs.box,
                             plotprefs=plotprefs)

'''
phase_diagram.contours()
phase_diagram.show()

phase_diagram.plotPressureColors()
phase_diagram.save(folderpath+"phase-diagram-points.png")
#phase_diagram.contours()
'''

phase_diagram.contoursPressure(levels=20, maxpressure=50)
phase_diagram.save(folderpath+"phase-diagram-contours-pressure.png")

phase_diagram.show()