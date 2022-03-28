from analysis.utils import load_history, PlotPreferences
import matplotlib.pyplot as plt
import numpy as np
from simulation.utils import UnitScaler

class EnergyPlotter:
    def __init__(self, filename, plotprefs=None, unitscaler=None, dimless=False):
        '''
        Class for plotting the energy evolution throughout a simulation

        :param filename: str or analysis.utils.History instance
                File or object containing the simulation data
        :param plotprefs: analysis.utils.PlotPreferences instance or NoneType
                If an instance of PlotPreferences, the given preferences are used.
                If None, the default layout is used. Default is None.
        :param unitscaler: simulation.utils.UnitScaler instance or NoneType
                If an instance of UnitScaler, uses the given scaler for converting to SI units.
                If NoneType, uses the default UnitScaler, for argon. Default is None.
        :param dimless: bool
                If True, plots dimensionless energies and times. Uses SI instead. Default is False.
        '''

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

        self.fig = plt.figure(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.ax = self.fig.add_subplot()

        if dimless:
            self.kin_energy = self.history.energies[:,1]
            self.pot_energy = self.history.energies[:,2]
            self.tot_energy = self.history.energies[:,3]
            self.times = self.history.energies[:,0]

            self.ax.set_xlabel("Time (natural units)")
            self.ax.set_ylabel("Energy (natural units)")

        else:
            self.kin_energy = self.unitscaler.toJoule(self.history.energies[:,1])
            self.pot_energy = self.unitscaler.toJoule(self.history.energies[:,2])
            self.tot_energy = self.unitscaler.toJoule(self.history.energies[:,3])
            self.times = self.unitscaler.toSeconds(self.history.energies[:,0])

            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Energy (J)")

        self.ax.set_yscale("linear")


    def plotKinetic(self, alpha=0.7):

        self.ax.plot(self.times, self.kin_energy, markersize=self.plotprefs.markersize,
                     label="Kinetic energy", alpha=alpha)

    def plotPotential(self, alpha=0.7):

        self.ax.plot(self.times, self.pot_energy, markersize=self.plotprefs.markersize,
                     label="Potential energy", alpha=alpha)

    def plotTotal(self, alpha=0.7):

        self.ax.plot(self.times, self.tot_energy, markersize=self.plotprefs.markersize,
                     label="Total energy", alpha=alpha, ls="--")

    def plotAll(self, alpha=0.7):

        self.plotKinetic(alpha=alpha)
        self.plotPotential(alpha=alpha)
        self.plotTotal(alpha=alpha)

    def show(self):

        self.ax.legend()
        self.fig.show()


class EquilibrationPlotter:

    def __init__(self, energy_fractions, plotprefs=None):
        '''
        Plot the fraction of target kinetic energy over measured kinetic energy throughout the equilibration process.

        :param energy_fractions: ndarray of shape(n_iterations)
                Fraction of target kinetic energy over measured kinetic energy for each equilibration step
        :param plotprefs: analysis.utils.PlotPreferences instance or NoneType
                If an instance of PlotPreferences, the given preferences are used.
                If None, the default layout is used. Default is None.
        '''

        self.energy_fractions = np.array(energy_fractions)
        self.iterations = np.array([i+1 for i in range(len(energy_fractions))])

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=5)

        else:
            self.plotprefs = plotprefs

        self.fig, self.ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.ax.set_xlabel(r"Iteration")
        self.ax.set_ylabel(r"$\frac{E_{kin,target}}{E_{kin,actual}}$")


    def plot(self):

        self.ax.plot(self.iterations, self.energy_fractions, markersize=self.plotprefs.markersize)


    def show(self):

        self.fig.show()