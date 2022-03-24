import numpy as np
from simulation.utils import LennardJonesPotential, minimumImagePositions, distanceSquaredFromPosition

class Particles:
    '''
    Class for handling a set of n_atoms particles.

    Parameters:
        n_atoms : int
            Number of particles
        dim : int
            Dimensions of the system
    '''

    def __init__(self, n_atoms, dim):
        self.n_atoms = n_atoms
        self.dim = dim

    def __len__(self):
        return self.n_atoms

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, pos):
        '''

        :param pos: ndarray of shape (self.n_atoms, self.dim) OR (self.dim, 2)
                    In the first case, taken as the positions of particles.
                    In the second case, taken as the edges of the box in which the particles live.
        '''

        if pos.shape == (self.n_atoms, self.dim):
            self._positions = pos

        elif pos.shape == (self.dim, 2):

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
        '''

        :param vel: ndarray of shape (self.n_atoms, self.dim) OR float
                    In the first case, takes input as velocities of particles.
                    In the second case, draws random velocities from a gaussian with given standard deviation.
        '''

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

        kinetic_energy = self.kineticEnergy()
        self._temperature = (2 / self.dim) * (kinetic_energy / (self.n_atoms - 1))

        return self._temperature

    @temperature.setter
    def temperature(self, inputTemp):
        '''Draws random particle velocities using the given dimensionless temperature.'''

        self.inputTemp = inputTemp
        self.velocities = np.sqrt(self.inputTemp)

        self._temperature = inputTemp


    def kineticEnergy(self):
        '''Computes the kinetic energy of the particles (in dimensionless units).'''

        kinetic_energy = 0.5 * np.sum(self.velocities**2)

        return kinetic_energy


    def potentialEnergy(self, box_length):
        '''
        Computes the potential energy of the system.
        :param box_length: float or ndarray of shape (self.dim,)
                    Indicates the linear size of the box
        '''

        potential = np.zeros(self.n_atoms)

        for i in range(self.n_atoms):
            pos = self.positions[i]
            # make sure not to count pairs twice
            pos_others = self.positions[i+1:]

            pos_others = minimumImagePositions(pos, pos_others, box_length)

            potential[i] = LennardJonesPotential(pos, pos_others)

        potential_energy = np.sum(potential)

        return potential_energy


    def pairDistances(self, box_length):
        '''Computes the distances between all pairs of particles in the simulation.

        :param box_length: float or ndarray of shape (self.dim,)

        '''

        distances2 = np.zeros((self.n_atoms, self.n_atoms-1))

        for i in range(self.n_atoms):

            pos = self.positions[i]
            pos_others = np.concatenate([self.positions[:i], self.positions[i+1:]])
            nearest_positions = minimumImagePositions(pos, pos_others, box_length)

            for j, other_pos in enumerate(nearest_positions):
                distances2[i,j] = distanceSquaredFromPosition(pos, other_pos)

        distances = np.sqrt(distances2)

        return distances


def initialiseLattice(unitlength, dim=3, units=3):
    '''
    Initialises particle positions on an FCC lattice.

    :param unitlength: linear size of an FCC base unit
    :param dim: dimensions of the system (2 or 3)
    :param units: number of unit cells to generate
    :return: positions on the lattice
    '''

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