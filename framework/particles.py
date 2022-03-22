import numpy as np
import h5py
from simulation.utils import LennardJonesPotential, minimumImagePositions, distanceSquaredFromPosition

class Particles:
    '''Class for handling a set of n_atoms particles.'''
    def __init__(self, n_atoms, dim, mass=1., unitscaler=None):
        self.n_atoms = n_atoms
        self.dim = dim
        self.mass = mass

        '''
        if unitscaler is None:
            self.unitscaler = UnitScaler()
        elif isinstance(unitscaler, UnitScaler):
            self.unitscaler = unitscaler
        else:
            raise TypeError("Invalid unitscaler given.")
            
        '''

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

            #position = np.random.uniform(low=pos[:,0], high=pos[:,1], size=(self.n_atoms, self.dim))
            #self._positions = position

            lengths = pos[:,1] - pos[:,0]
            n_units = int((self.n_atoms / (self.dim + 1))**(1/self.dim))
            unitlength = lengths[0] / 3

            self._positions = initialiseLattice(unitlength, self.dim, n_units)


        else:
            print ("Invalid argument given.")
            print ("Input should be either an array of positions of shape (n_atoms, dim),")
            print ("Or an array of box edges of shape (dim, 2).")


    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, vel):
        if isinstance(vel, np.ndarray):
            if vel.shape == (self.n_atoms, self.dim):
                self._velocities = vel
                #print ("Velocities set.")

            else:
                print ("Given velocities have incorrect dimensions.")

        else:
            # the user will give a standard deviation for the gaussian
            # generate random positions from a gaussian
            mean = 0.
            scale = vel
            self._velocities = np.random.normal(mean, scale, size=(self.n_atoms, self.dim))


    @property
    def temperature(self):
        '''Calculate the actual temperature given the particles' velocities.'''

        self._temperature = np.mean(self.velocities**2)

        return self._temperature

    @temperature.setter
    def temperature(self, dimlessTemp):
        '''Draws random particle velocities using the given real temperature in K.'''

        self.dimlessTemp = dimlessTemp
        self.velocities = np.sqrt(self.dimlessTemp)

        self._temperature = dimlessTemp


    def kineticEnergy(self):
        '''Computes the kinetic energy of the particles (in dimensionless units).'''

        kinetic_energy = 0.5 * np.sum(self.velocities**2)

        return kinetic_energy


    def potentialEnergy(self, box_length):

        potential = np.zeros(self.n_atoms)

        for i in range(self.n_atoms):
            pos = self.positions[i]
            # make sure not to count pairs twice
            pos_others = self.positions[i+1:]

            pos_others = minimumImagePositions(pos, pos_others, box_length)

            '''
            # use the minimum image convention
            pos_diff = pos_others - pos
            pos_others = pos_others - box_length * np.rint(pos_diff/box_length)
            '''
            potential[i] = LennardJonesPotential(pos, pos_others)

        potential_energy = np.sum(potential)

        return potential_energy


    def pairDistances(self, box_length):
        '''Computes the distances between all pairs of particles in the simulation.'''

        distances2 = np.zeros((self.n_atoms, self.n_atoms-1))

        for i in range(self.n_atoms):

            pos = self.positions[i]
            pos_others = np.concatenate([self.positions[:i], self.positions[i+1:]])
            nearest_positions = minimumImagePositions(pos, pos_others, box_length)

            for j, other_pos in enumerate(nearest_positions):
                distances2[i,j] = distanceSquaredFromPosition(pos, other_pos)

        distances = np.sqrt(distances2)

        return distances


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


def initialiseLattice(unitlength, dim=3, units=3):

    positions = []

    if dim == 2:
        for i in range(units):
            for j in range(units):
                positions.append([i,j])
                positions.append([i,j+0.5])
                positions.append([i+0.5,j])

    elif dim == 3:
        for i in range(units):
            for j in range(units):
                for k in range(units):
                    positions.append([i,j,k])
                    positions.append([i,j,k+0.5])
                    positions.append([i,j+0.5,k])
                    positions.append([i+0.5,j,k])


    return np.array(positions) * unitlength