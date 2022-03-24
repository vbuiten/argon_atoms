''''Module for computing the pressure in a system of particles.'''

import numpy as np
from simulation.utils import distanceSquaredFromPosition, minimumImagePositions
from framework.particles import Particles
from analysis.utils import RepeatedSimsBase, PlotPreferences, VaryingInitialConditionsSims
import matplotlib.pyplot as plt

def pressureFromParticles(particles_list, box_lengths, density, temperature):

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

    '''
    prefactor1 = temperature * density
    prefactor2 = 4. / (particles_list[0].n_atoms * temperature)
    '''

    print (prefactor1)
    print (prefactor2)
    print (potential_terms)

    pressure = prefactor1 * (1 - prefactor2 * potential_terms)
    print (pressure)

    return pressure


class VirialPressure(RepeatedSimsBase):
    def __init__(self, particles_list, box_lengths, plotprefs=None):
        super().__init__(particles_list, box_lengths)

        '''
        potential_terms = np.zeros(len(particles))

        for i, set in enumerate(particles):

            terms_particles = np.zeros(len(set.positions))

            for idx, position in enumerate(set.positions):

                pos_others = minimumImagePositions(position, set.positions[idx+1:], box_lengths)

                distances2 = np.zeros(len(pos_others))
                for j in range(len(pos_others)):
                    distances2[j] = distanceSquaredFromPosition(position, pos_others[j])

                terms_particles[idx] = np.sum(distances2**(-3) - 2 * distances2**(-6))

            potential_terms[i] = np.sum(terms_particles)

        prefactor1 = self.temperature * self.density
        prefactor2 = 4. / (self.n_atoms * self.temperature)

        self.pressures = prefactor1 * (1 - prefactor2 * potential_terms)
        '''

        self.pressures = pressureFromParticles(self.particles, box_lengths, self.density, self.temperature)

        self.pressure_avg = np.mean(self.pressures)
        self.pressure_68p = np.percentile(self.pressures, [16.,84.])

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
        super().__init__(particles, boxes)

        self.pressures = pressureFromParticles(particles, self.box_lengths, self.density, self.temperature)

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=3, marker="s")
        else:
            self.plotprefs = plotprefs

        self.fig, self.ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)
        self.fig.suptitle("Phase Diagram")


    def plot(self):

        sc = self.ax.scatter(self.temperature, self.pressures, c=self.density, s=self.plotprefs.markersize,
                          marker=self.plotprefs.marker)
        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Pressure")
        cbar = self.fig.colorbar(sc, label="Density")


    def contours(self):

        cntr = self.ax.tricontourf(self.temperature, self.pressures, self.density)
        self.ax.set_xlabel("Temperature")
        self.ax.set_ylabel("Pressure")
        cbar = self.fig.colorbar(cntr, label="Density")


    def show(self):

        self.fig.show()

    def save(self, savefile):

        self.fig.savefig(savefile, bbox_inches="tight")