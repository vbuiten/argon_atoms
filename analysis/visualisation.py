'''Code for visualising the simulation data (e.g. trajectory plots).'''

import numpy as np
import matplotlib.pyplot as plt

import analysis.utils
from analysis.utils import load_history, PlotPreferences

class TrajectoryPlotter:
    def __init__(self, history, plotprefs=None):

        if plotprefs is None:
            self.plotprefs = PlotPreferences()

        else:
            self.plotprefs = plotprefs

        self.history = load_history(history)

        box_edges = self.history.box_edges

        self.fig = plt.figure(figsize=self.plotprefs.figsize, dpi=self.plotprefs.dpi)

        if self.history.dim == 2:
            self.ax = self.fig.add_subplot()
            self.plot3D = False

        elif self.history.dim == 3:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_zlabel("z")
            self.ax.set_zlim(box_edges[2,0], box_edges[2,1])
            self.plot3D = True

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_aspect("equal")

        self.ax.set_xlim(box_edges[0,0], box_edges[0,1])
        self.ax.set_ylim(box_edges[1,0], box_edges[1,1])

    def scatter(self, min_idx, max_idx):

        for i in range(self.history.n_atoms):
            trajectory = self.history.pos[min_idx:max_idx,i]

            if self.plot3D:
                self.ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2], alpha=0.5,
                                markersize=self.plotprefs.markersize)

            else:
                self.ax.scatter(trajectory[:,0], trajectory[:,1], alpha=0.5, markersize=self.plotprefs.markersize)

    def plot(self, min_idx, max_idx):

        for i in range(self.history.n_atoms):
            trajectory = self.history.pos[min_idx:max_idx,i]

            if self.plot3D:
                self.ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], alpha=0.3, marker="o",
                             markersize=self.plotprefs.markersize)

            else:
                self.ax.plot(trajectory[:,0], trajectory[:,1], alpha=0.3, marker="o",
                             markersize=self.plotprefs.markersize)

    def show(self):

        self.fig.show()