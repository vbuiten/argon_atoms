import numpy as np

class Particles:
    '''Class for handling a set of n_atoms particles.'''
    def __init__(self, n_atoms, dim):
        self.n_atoms = n_atoms
        self.dim = dim

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, pos):
        if isinstance(pos, np.ndarray):
            self._positions = pos

        elif pos=="random":
            # generate random positions from a uniform distribution
            # need to allow the user to set limits
            # through the Box object?

            position = np.random.uniform(low=0, high=1, size=(self.n_atoms, self.dim))
            self._positions = position