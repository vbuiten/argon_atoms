import numpy as np
from scipy.stats import multivariate_normal
import h5py
from simulation.utils import LennardJonesPotential, UnitScaler

class Particles:
    '''Class for handling a set of n_atoms particles.'''
    def __init__(self, n_atoms, dim, mass=1., unitscaler=None):
        self.n_atoms = n_atoms
        self.dim = dim
        self.mass = mass

        if unitscaler is None:
            self.unitscaler = UnitScaler()
        elif isinstance(unitscaler, UnitScaler):
            self.unitscaler = unitscaler
        else:
            raise TypeError("Invalid unitscaler given.")

        self.savefile = None

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


    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, dimlessTemp):
        '''Draws random particle velocities using the given real temperature in K.'''

        #self._temperature = temp
        #self.dimlessTemp = self.unitscaler.toDimlessTemperature(temp)
        self.dimlessTemp = dimlessTemp
        mean = np.zeros(self.dim)
        cov = np.diag(np.sqrt(self.dimlessTemp) * np.ones(self.dim))
        print (cov)
        gauss = multivariate_normal(mean=mean, cov=cov)
        self._velocities = gauss.rvs(size=(self.n_atoms))

        self._temperature = self.unitscaler.toKelvinFromDimlessTemperature(self.dimlessTemp)


    def kineticEnergy(self):
        '''Computes the kinetic energy of the particles (in dimensionless units).'''

        velsquared = np.zeros(self.n_atoms)
        for i in range(self.n_atoms):
            velsquared[i] = np.dot(self.velocities[i], self.velocities[i])

        kinetic_energy = 0.5 * np.sum(velsquared)

        return kinetic_energy


    def potentialEnergy(self, box_length):

        potential = np.zeros(self.n_atoms)

        for i in range(self.n_atoms):
            pos = self.positions[i]
            # make sure not to count pairs twice
            pos_others = self.positions[i+1:]

            # use the minimum image convention
            pos_diff = pos_others - pos
            pos_others = pos_others - box_length * np.rint(pos_diff/box_length)
            potential[i] = LennardJonesPotential(pos, pos_others)

        potential_energy = np.sum(potential)

        return potential_energy

    def createFile(self, savefile):
        self.savefile = savefile
        file = h5py.File(savefile, "w")
        file.create_group("positions")
        file.create_group("velocities")
        #datasets = file.create_dataset("particles", )
        file.close()

    def saveToFile(self, savefile=None):

        if self.savefile is None:
            self.createFile(savefile)

        file = h5py.File(self.savefile, "r+")
        dataset = file["/particles"]
        dataset["positions"].append(self.positions)
