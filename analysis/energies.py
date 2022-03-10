from analysis.utils import load_history, PlotPreferences
import matplotlib.pyplot as plt
from simulation.utils import UnitScaler

class EnergyPlotter:
    def __init__(self, filename, plotprefs=None, unitscaler=None):

        self.history = load_history(filename)

        if plotprefs is None:
            self.plotprefs = PlotPreferences()

        else:
            self.plotprefs = plotprefs

        if unitscaler is None:
            self.unitscaler = UnitScaler()
        elif isinstance(unitscaler, UnitScaler):
            self.unitscaler = unitscaler
        else:
            raise TypeError("Invalid unitscaler given.")

        self.history = load_history(filename)
        self.kin_energy = self.unitscaler.toJoule(self.history.energies[:,1])
        self.pot_energy = self.unitscaler.toJoule(self.history.energies[:,2])
        self.tot_energy = self.unitscaler.toJoule(self.history.energies[:,3])
        self.times = self.unitscaler.toSeconds(self.history.energies[:,0])

        self.fig = plt.figure(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.ax = self.fig.add_subplot()

        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Energy (J)")


    def plotKinetic(self, alpha=0.7):

        self.ax.plot(self.times, self.kin_energy, markersize=self.plotprefs.markersize,
                     label="Kinetic energy", alpha=alpha)

    def plotPotential(self, alpha=0.7):

        self.ax.plot(self.times, self.pot_energy, markersize=self.plotprefs.markersize,
                     label="Potential energy", alpha=alpha)

    def plotTotal(self, alpha=0.7):

        self.ax.plot(self.times, self.tot_energy, markersize=self.plotprefs.markersize,
                     label="Total energy", alpha=alpha)

    def plotAll(self, alpha=0.7):

        self.plotKinetic(alpha=alpha)
        self.plotPotential(alpha=alpha)
        self.plotTotal(alpha=alpha)

    def show(self):

        self.ax.legend()
        self.fig.show()