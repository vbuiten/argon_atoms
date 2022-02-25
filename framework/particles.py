import numpy as np
from scipy.stats import multivariate_normal

class Particles:
    '''Class for handling a set of n_atoms particles.'''
    def __init__(self, n_atoms, dim, mass=1.):
        self.n_atoms = n_atoms
        self.dim = dim
        self.mass = mass

    def __len__(self):
        return self.n_atoms

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, pos):
        if pos.shape == (self.n_atoms, self.dim):
            self._positions = pos

        elif pos.shape == (self.dim, 2):
            # generate random positions from a uniform distribution
            # pos can be passed as an array containing the edges of the box

            position = np.random.uniform(low=pos[:,0], high=pos[:,1], size=(self.n_atoms, self.dim))
            self._positions = position

        else:
            print ("Invalid argument given.")
            print ("Input should be either an array of positions of shape (n_atoms, dim),")
            print ("Or an array of box edges of shape (dim, 2).")


    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, vel):
        try:
            if vel.shape == (self.n_atoms, self.dim):
                self._velocities = vel

            else:
                print ("Given velocities have incorrent dimensions.")

        except:
            # the user will give a standard deviation for the gaussian
            # generate random positions from a gaussian
            # we'll probably want to change this to a Maxwell-Boltzmann distribution
            mean = np.zeros(self.dim)
            cov = np.diag(vel * np.ones(self.dim))
            gauss = multivariate_normal(mean=mean, cov=cov)
            self._velocities = gauss.rvs(size=(self.n_atoms))


    def kineticEnergy(self):
        '''Computes the kinetic energy of the particles (in dimensionless units).'''

        velsquared = np.zeros(self.n_atoms)
        for i in range(self.n_atoms):
            velsquared[i] = np.dot(self.velocities[i], self.velocities[i])

        kinetic_energy = 0.5 * np.sum(velsquared)

        return kinetic_energy