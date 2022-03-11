import numpy as np

class BoxBase:
    '''Class setting up the box in which the particles live.'''

    def __init__(self, dimlessDensity, n_atoms, dim=3):

        self.dim = dim

        # set the dimensionless volume and box lengths
        self.volume = n_atoms / dimlessDensity
        length = self.volume**(1./self.dim)
        self.lengths = length * np.ones(self.dim)

        edges = np.zeros((self.dim,2))
        edges[:,1] = self.lengths
        self.edges = edges