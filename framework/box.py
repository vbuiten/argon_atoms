import numpy as np

class BoxBase:
    '''

    Class setting up the square/cubic box in which the particles live.
    Each side of the box spans from 0 to length.

    Parameters:
        dimlessDensity : float
            Desired dimensionless density of particles in the box.

        n_atoms : int
            Desired number of atoms living in the box.

        dim : int
            Dimensions of the box. Default is 3; 2 can be used for tests but will not provide accurate science results.

    '''

    def __init__(self, dimlessDensity, n_atoms, dim=3):

        self.dim = dim

        # set the dimensionless volume and box lengths
        self.density = dimlessDensity
        self.volume = n_atoms / dimlessDensity
        length = self.volume**(1./self.dim)
        self.lengths = length * np.ones(self.dim)
        self.length = length

        edges = np.zeros((self.dim,2))
        edges[:,1] = self.lengths
        self.edges = edges