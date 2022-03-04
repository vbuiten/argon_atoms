'''Code for visualising the simulation data (e.g. trajectory plots).'''

import numpy as np
import matplotlib.pyplot as plt

import analysis.utils
from analysis.utils import History

class TrajectoryPlotter:
    def __init__(self, history):

        if isinstance(history, analysis.utils.History):
            self.history = history
        elif isinstance(history, str):
            self.history = History(filename=history)

        self.fig = plt.figure()

        if self.history.dim == 2:
            self.ax = self.fig.add_subplot()
            self.plot3D = False

        elif self.history.dim == 3:
            self.ax = self.fig.add_subplot(projection="3d")
            self.ax.set_zlabel("z")
            self.plot3D = True

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def scatter(self, min_idx, max_idx):

        for i in range(self.history.n_atoms):
            trajectory = self.history.pos[min_idx:max_idx,i]

            if self.plot3D:
                self.ax.scatter(trajectory[0], trajectory[1], trajectory[2], alpha=0.5)

            else:
                self.ax.scatter(trajectory[0]. trajectory[1], alpha=0.5)

    def show(self):

        self.fig.show()