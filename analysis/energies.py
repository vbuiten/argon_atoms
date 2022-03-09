from analysis.utils import load_history, PlotPreferences
import matplotlib.pyplot as plt

class EnergyPlotter:
    def __init__(self, filename, plotprefs=None):

        self.history = load_history(filename)

        if plotprefs is None:
            self.plotprefs = PlotPreferences()

        else:
            self.plotprefs = plotprefs

        self.history = load_history(filename)

        self.fig = plt.figure(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.ax = self.fig.add_subplot()

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Energy")


    def plotKinetic(self, alpha=0.7):

        self.ax.plot(self.history.times, self.history.energies[:,1], markersize=self.plotprefs.markersize,
                     label="Kinetic energy", alpha=alpha)

    def plotPotential(self, alpha=0.7):

        self.ax.plot(self.history.times, self.history.energies[:,2], markersize=self.plotprefs.markersize,
                     label="Potential energy", alpha=alpha)

    def plotTotal(self, alpha=0.7):

        self.ax.plot(self.history.times, self.history.energies[:,3], markersize=self.plotprefs.markersize,
                     label="Total energy", alpha=alpha)

    def plotAll(self, alpha=0.7):

        self.plotKinetic(alpha=alpha)
        self.plotPotential(alpha=alpha)
        self.plotTotal(alpha=alpha)

    def show(self):

        self.ax.legend()
        self.fig.show()