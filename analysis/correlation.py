'''Module for measuring pair correlation'''

import numpy as np
import matplotlib.pyplot as plt

import framework.particles
from simulation.utils import minimumImagePositions
from analysis.utils import PlotPreferences

class DistanceHistogram:
    '''Create a histogram of the distances between pairs of particles.'''

    def __init__(self, particles, box_lengths, bins, plotprefs=None):

        if isinstance(particles, framework.particles.Particles):
            particles = [particles]

        n_atoms = particles[0].n_atoms

        distances = np.array((len(particles), n_atoms, n_atoms-1))

        for idx, el in enumerate(particles):

            distances[idx] = el.pairDistances(box_lengths)

        self.counts, self.bin_edges = np.histogram(distances, bins)
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.bin_mids = self.bin_edges[:-1] + 0.5*self.bin_widths

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=5, marker="s")


    def plot(self):

        fig, ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)

        ax.plot(self.bin_mids, self.counts, marker=self.plotprefs.marker)
        ax.set_xlabel(r"Distance")
        ax.set_ylabel(r"# of occurrences")

        fig.suptitle(r"Pair Correlation Function $g(r)$")

        self.fig = fig
        self.ax = ax


    def show(self):

        self.fig.show()