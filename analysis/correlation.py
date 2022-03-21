'''Module for measuring pair correlation'''

import numpy as np
import matplotlib.pyplot as plt

import framework.particles
from simulation.utils import minimumImagePositions
from analysis.utils import PlotPreferences, RepeatedSimsBase

class DistanceHistogram(RepeatedSimsBase):
    '''Create a histogram of the distances between pairs of particles.'''

    def __init__(self, particles, box_lengths, bins=10, plotprefs=None):

        super().__init__(particles, box_lengths)

        self.distances = np.zeros((len(particles), self.n_atoms, self.n_atoms-1))

        for idx, el in enumerate(particles):

            self.distances[idx] = el.pairDistances(box_lengths)

        if isinstance(bins, int):
            min_edge, max_edge = 0., self.distances.max()
            width = (max_edge - min_edge)/bins
            bin_edges = np.arange(min_edge, max_edge, width)

        elif isinstance(bins, np.ndarray):
            bin_edges = bins

        else:
            raise TypeError("Invalid bins. Give either the number of bins or the edges.")

        self.counts, self.bin_edges = np.histogram(self.distances, bin_edges)
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.bin_mids = self.bin_edges[:-1] + 0.5*self.bin_widths

        counts_iterations = np.zeros((self.n_iterations, len(self.bin_mids)))

        for i in range(self.n_iterations):
            counts_iterations[i], _ = np.histogram(self.distances[i], bin_edges)

        self.counts_avg = np.mean(counts_iterations, axis=0)
        self.counts_68p = np.percentile(counts_iterations, [16.,84], axis=0)
        self.counts_errs = np.abs(self.counts_avg - self.counts_68p)
        self.counts_std = np.std(counts_iterations, axis=0)

        if plotprefs is None:
            self.plotprefs = PlotPreferences(markersize=5, marker="s")

        self.fig, self.ax = plt.subplots(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)

        self.ax.set_xlabel(r"Distance")
        self.ax.set_ylabel(r"# of occurrences")


    def plotTotal(self):

        self.ax.plot(self.bin_mids, self.counts, marker=self.plotprefs.marker)


    def plotIterationAveraged(self):

        self.ax.plot(self.bin_mids, self.counts_avg, markersize=self.plotprefs.markersize, marker=self.plotprefs.marker,
                     label="Mean")
        self.ax.fill_between(self.bin_mids, self.counts_68p[0], self.counts_68p[1], alpha=0.3,
                             label="68% confidence interval")

        self.ax.set_title("Averaged over "+str(self.n_iterations)+" Realisations")
        self.ax.legend()


    def show(self):

        self.fig.show()


class CorrelationFunction(DistanceHistogram):
    def __init__(self, particles, box_lengths, bins=10, plotprefs=None):
        super().__init__(particles, box_lengths, bins, plotprefs)

        factor1 = 2 * self.volume / (self.n_atoms * (self.n_atoms - 1))
        factor2 = 1 / (4 * np.pi * self.bin_mids**2 * self.bin_widths)
        self.g = factor1 * factor2 * self.counts_avg

        self.g_error = factor1 * factor2 * self.counts_std

        self.ax.set_ylabel(r"Correlation Function $g(r)$")
        self.fig.suptitle("Correlation Function")
        self.ax.set_title(r"$\rho = $"+str(np.around(self.density,3))+r"; $T = $"+str(np.around(self.temperature,3)))

    def plot(self):

        self.ax.plot(self.bin_mids, self.g, marker=self.plotprefs.marker, markersize=self.plotprefs.markersize)
        self.ax.fill_between(self.bin_mids, self.g+self.g_error, self.g-self.g_error, alpha=0.3,
                             label=r"Estimated $1 \sigma$ error")

    def show(self):

        self.fig.show()