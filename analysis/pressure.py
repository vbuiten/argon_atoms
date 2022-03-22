''''Module for computing the pressure in a system of particles.'''

import numpy as np
from simulation.utils import distanceSquaredFromPosition, minimumImagePositions
from framework.particles import Particles
from analysis.utils import RepeatedSimsBase, PlotPreferences
import matplotlib.pyplot as plt

class VirialPressure(RepeatedSimsBase):
    def __init__(self, particles, box_lengths, plotprefs=None):
        super().__init__(particles, box_lengths)

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