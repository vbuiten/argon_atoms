''''Module for computing the pressure in a system of particles.'''

import numpy as np
from simulation.utils import distanceSquaredFromPosition, minimumImagePositions
from framework.particles import Particles
from analysis.utils import RepeatedSimsBase, PlotPreferences, VaryingInitialConditionsSims
import matplotlib.pyplot as plt

def pressureFromParticles(particles_list, box_lengths, density):
    '''
    Calculates the pressure for each system in particles_list.

    :param particles_list: list of framework.particles.Particles instances or of analysis.utils.ParticlesFromHistory instances
            List containing all particle systems to consider
    :param box_lengths: float or ndarray of shape (n_sims, dim) or (n_sims,) or (dim,) if box length is the same for all systems
    :param density: float or ndarray of shape (n_sims)
            Density given to each system
    :return: pressure: ndarray of shape (n_sims)
            Calculated pressure for each system of particles given
    '''


    potential_terms = np.zeros(len(particles_list))
    temperatures = np.zeros(len(particles_list))

    for i, set in enumerate(particles_list):

        terms_set_i = np.zeros(len(set.positions))
        temperatures[i] = set.temperature

        for idx, position in enumerate(set.positions):

            try:
                pos_others = minimumImagePositions(position, set.positions[idx+1:], box_lengths)
            except:
                pos_others = minimumImagePositions(position, set.positions[idx+1:], box_lengths[i])

            distances2 = np.zeros(len(pos_others))
            for j in range(len(pos_others)):
                distances2[j] = distanceSquaredFromPosition(position, pos_others[j])

            terms_set_i[idx] = np.sum(distances2 ** (-3) - 2 * distances2 ** (-6))

        potential_terms[i] = np.sum(terms_set_i)

    prefactor1 = temperatures * density
    prefactor2 = 4. / (particles_list[0].n_atoms * temperatures)

    pressure = prefactor1 * (1 - prefactor2 * potential_terms)

    return pressure


class VirialPressure(RepeatedSimsBase):
    '''
    Class for calculating and plotting the pressure for each system of particles provided.

    :param particles_list: list of analysis.utils.ParticlesFromHistory instances
            List of particle systems to analyse
    :param box_lengths: ndarray of shape (n_sims, dim)
            Linear sizes of the box used for each simulation
    :param plotprefs: NoneType or analysis.utils.PlotPreferences instance
            If an instance of PlotPreferences, the given preferences are used.
            If None, the default layout is used.
    '''

    def __init__(self, particles_list, box_lengths, plotprefs=None):
        super().__init__(particles_list, box_lengths)

        self.pressures = pressureFromParticles(self.particles, box_lengths, self.density, self.temperature)

        self.pressure_avg = np.mean(self.pressures)
        self.pressure_68p = np.percentile(self.pressures, [16.,84.])

        print ("Mean pressure:", self.pressure_avg)
        print ("Pressure errors:", self.pressure_avg-self.pressure_68p)

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=3, marker="s")
        else:
            self.plotprefs = plotprefs

        self.fig, self.ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.fig.suptitle("Pressure Measurements")

    def plot(self):

        self.ax.hist(self.pressures, alpha=0.7, bins="sqrt")
        self.ax.axvline(self.pressure_avg, ls="--", color="black", label="Mean")
        self.ax.axvline(self.pressure_68p[0], ls=":", color="black", label="68\% interval")
        self.ax.axvline(self.pressure_68p[1], ls=":", color="black")

        self.ax.legend()
        self.ax.set_xlabel("Dimensionless pressure")
        self.ax.set_ylabel("Occurrences")

        self.ax.set_title(r"$\rho = $"+str(np.around(self.density,2))+r"; $T = $"+str(np.around(self.temperature,2)))

    def show(self):

        self.fig.show()


    def save(self, savefile):

        self.fig.savefig(savefile, bbox_inches="tight")


class PhaseDiagram(VaryingInitialConditionsSims):

    def __init__(self, particles, boxes, plotprefs=None):
        '''
        Class for plotting a phase diagram of all systems in particles.

        :param particles: list of analysis.utils.ParticlesFromHistory instances
                Particle systems to plot
        :param boxes: list of framework.box.BoxBase instances
                Boxes corresponding to each particle system.
        :param plotprefs: analysis.utils.PlotPreferences instance or NoneType
                If an instance of PlotPreferences, the given preferences are used.
                If None, the default layout is used. Default is None.
        '''

        super().__init__(particles, boxes)

        self.pressures = pressureFromParticles(particles, self.box_lengths, self.density)

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=3, marker="s")
        else:
            self.plotprefs = plotprefs

        self.fig, self.ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.fig.suptitle("Phase Diagram")


    def plot(self):
        '''
        Make a scatter plot of temperature vs. pressure, with points coloured according to their density.
        '''

        sc = self.ax.scatter(self.temperature, self.pressures, c=self.density, s=self.plotprefs.markersize,
                          marker=self.plotprefs.marker)
        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Pressure")
        cbar = self.fig.colorbar(sc, label="Density")


    def plotPressureColors(self):
        '''
        Make a scatter plot of temperature vs. density, with points coloured according to pressure.
        '''

        sc = self.ax.scatter(self.temperature, self.density, c=self.pressures, s=self.plotprefs.markersize,
                             marker=self.plotprefs.marker, vmax=50)
        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Density")
        cbar = self.fig.colorbar(sc, label="Pressure")


    def contours(self):
        '''
        Make a contour plot of temperature vs. pressure with contours for the density.
        '''

        cntr = self.ax.tricontourf(self.temperature, self.pressures, self.density)
        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Pressure")
        cbar = self.fig.colorbar(cntr, label="Density")


    def contoursPressure(self, levels=15, minpressure=0.0001, maxpressure=1000):
        '''
        Make a contour plot of temperature vs. density with contours for the pressure.

        :param levels: int
                Number of contour levels to draw
        :param minpressure: float
                Minimum pressure to include
        :param maxpressure: float
                Maximum pressure to include
        '''

        goodpressure = (self.pressures < maxpressure) & (self.pressures > minpressure)
        print ("Number of points used:", np.sum(goodpressure))

        self.ax.tricontour(self.temperature[goodpressure], self.density[goodpressure],
                           self.pressures[goodpressure], levels=levels, colors="white",
                           linewidths=0.3)
        cntr = self.ax.tricontourf(self.temperature[goodpressure], self.density[goodpressure],
                                   self.pressures[goodpressure], levels=levels,
                                   cmap="turbo")
        sc = self.ax.scatter(self.temperature[goodpressure], self.density[goodpressure], c="k", s=self.plotprefs.markersize,
                             marker=self.plotprefs.marker, vmax=50)

        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Density")
        self.ax.set_title(str(np.sum(goodpressure))+" measurements with $P < $ "+str(maxpressure))
        cbar = self.fig.colorbar(cntr, label="Pressure")

    def show(self):

        self.fig.show()

    def save(self, savefile):

        self.fig.savefig(savefile, bbox_inches="tight")