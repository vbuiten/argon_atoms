'''Test file for experimenting with analysing several realisations of the same initial conditions.'''

from framework.box import BoxBase
from framework.particles import Particles
from simulation.sim import Simulator
from analysis.correlation import DistanceHistogram, CorrelationFunction
from analysis.pressure import VirialPressure
from analysis.utils import PlotPreferences, SimulationIterations
import os

savepath = "C:\\Users\\victo\\Documents\\Uni\\COP\\"

if not os.access(savepath, os.F_OK):
    savepath = "/net/vdesk/data2/buiten/COP/"

    foldername = "/dens0.3-temp3.0/"
    folderpath = savepath+foldername

else:
    foldername = "\\dens0.3-temp3.0\\"
    folderpath = savepath+foldername

if not os.access(folderpath, os.F_OK):
    os.mkdir(folderpath)

plotprefs = PlotPreferences(usetex=True, markersize=3, marker="o")

sim_runs = SimulationIterations(folderpath)

distance_hist = DistanceHistogram(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
distance_hist.plotIterationAveraged()
distance_hist.save(folderpath+"disthist.png")
distance_hist.show()

corr_func = CorrelationFunction(sim_runs.final_particles, sim_runs.box.lengths, bins=100, plotprefs=plotprefs)
corr_func.plot()
corr_func.save(folderpath+"corrfunc.png")
corr_func.show()

pressure = VirialPressure(sim_runs.final_particles, sim_runs.box.lengths, plotprefs=plotprefs)
pressure.plot()
pressure.save(folderpath+"pressure.png")
pressure.show()