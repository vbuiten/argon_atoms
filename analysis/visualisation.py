'''Code for visualising the simulation data (e.g. trajectory plots).'''

import numpy as np
import matplotlib.pyplot as plt

import analysis.utils
from analysis.utils import load_history, PlotPreferences

class TrajectoryPlotter:
    '''
    Class for making plots of particle trajectories in a simulation.

    :param history: str or analysis.utils.History instance
            File or History instance containing the simulation data
    :param plotprefs: NoneType or analysis.utils.PlotPreferences instance
            If an instance of PlotPreferences, the given preferences are used.
            If None, the default layout is used.
    '''

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
            self.ax.set_aspect("equal")

        elif self.history.dim == 3:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_zlabel("z")
            self.ax.set_zlim(box_edges[2,0], box_edges[2,1])
            self.plot3D = True
            self.ax.set_aspect("auto")

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.ax.set_xlim(box_edges[0,0], box_edges[0,1])
        self.ax.set_ylim(box_edges[1,0], box_edges[1,1])

    def scatter(self, min_idx, max_idx):
        '''
        Makes a scatter plot of trajectories.

        :param min_idx: int
                Index of the first time step to plot
        :param max_idx: int
                Index up to which the time steps are shown (exclusive)
        '''

        for i in range(self.history.n_atoms):
            trajectory = self.history.pos[min_idx:max_idx,i]

            if self.plot3D:
                self.ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2], alpha=0.5,
                                s=self.plotprefs.markersize)

            else:
                self.ax.scatter(trajectory[:,0], trajectory[:,1], alpha=0.5, s=self.plotprefs.markersize)


    def plot(self, min_idx, max_idx):
        '''
        Makes a plot with lines between points in time for each particle.

        :param min_idx: int
                Index of the first time step to plot
        :param max_idx: int
                Index up to which the time steps are shown (exclusive)
        '''

        for i in range(self.history.n_atoms):
            trajectory = self.history.pos[min_idx:max_idx,i]

            if self.plot3D:
                self.ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], alpha=0.3, marker=".",
                             markersize=self.plotprefs.markersize, color="tab:blue")
                self.ax.plot(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], alpha=0.5, marker="o",
                             markersize=self.plotprefs.markersize)

            else:
                self.ax.plot(trajectory[:,0], trajectory[:,1], alpha=0.3, marker=".",
                             markersize=self.plotprefs.markersize, color="tab:blue")
                self.ax.plot(trajectory[-1, 0], trajectory[-1, 1], alpha=0.5, marker="o",
                             markersize=self.plotprefs.markersize)

    def show(self):
        '''
        Show the figure.
        '''

        self.fig.show()