import numpy as np

class BoxBase:
    '''Class setting up the square/cubic box in which the particles live.
    Each side of the box spans from 0 to length.'''

    def __init__(self, dimlessDensity, n_atoms, dim=3):

        self.dim = dim

        # set the dimensionless volume and box lengths
        self.volume = n_atoms / dimlessDensity
        length = self.volume**(1./self.dim)
        self.lengths = length * np.ones(self.dim)
        self.length = length

        edges = np.zeros((self.dim,2))
        edges[:,1] = self.lengths
        self.edges = edges