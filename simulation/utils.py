'''Module containing utility functions for doing argon simulations.'''

import numpy as np
from numba import jit

@jit(nopython=True, parallel=False)
def distanceSquaredFromPosition(pos1, pos2):
    '''
    Calculates the distance squared between 1D arrays pos1 and pos2.

    :param pos1: ndarray of shape (dim,) (i.e. (2,) or (3,)
                Cartesian coordinates of particle 1
    :param pos2: ndarray of shape (dim,)
                Cartesian coordinates of particle 2
    :return dist: float
                Distance between the two particles
    '''

    diffVector = pos2 - pos1
    dist = np.dot(diffVector, diffVector)

    return dist

@jit(nopython=True, parallel=False)
def posInBox(pos, lengths):
    '''
    Simple function for shifting a particle position inside the box.

    :param pos: ndarray of shape (n_particles, dim)
                Cartesian coordinates of particles
    :param lengths: float or ndarray of shape (dim,)
                Specifies the linear size of the box
    :return pos: ndarray of shape (n_particles, dim)
                Shifted particle coordinates, now all inside the box under periodic boundary conditions
    '''

    pos = (pos + 2*lengths) % lengths
    return pos


@jit(nopython=True)
def minimumImagePositions(position, pos_others, lengths):
    '''
    Gives the positions of the nearest periodic copy of each particle in the interaction.

    :param position: ndarray of shape (dim,)
                Cartesian coordinates of the particle for which the interactions are considered
    :param pos_others: ndarray of shape (n_other_particles, dim)
                Cartesian coordinates of all particles that interact with the particle under consideration
    :param lengths: float or ndarray of shape (dim,)
                Linear size(s) of the box
    :return: nearest_positions: ndarray of shape (n_other_particles, dim)
                Cartesian coordinates of the periodic copies of each particle closest to the particle in question
    '''

    pos_diff = position - pos_others
    nearest_positions = pos_others + lengths * np.rint(pos_diff/lengths)

    return nearest_positions


@jit(nopython=True, parallel=False)
def minimumImageForces(positions, lengths):
    '''
    Calculates the Lennard-Jones forces acting on each particle under the minimum image convention.

    :param positions: ndarray of shape (n_particles, dim)
                Coordinates of each particle in the system
    :param lengths: float or ndarray of shape (dim,)
                Linear size(s) of the box
    :return: forces: ndarray of shape (n_particles, dim)
                Cartesian components of the net force acting on each particle
    '''

    forces = np.zeros(positions.shape)

    for i in range(len(forces)):
        pos = positions[i]
        pos_others = np.concatenate((positions[:i], positions[i+1:]))

        nearest_positions = minimumImagePositions(pos, pos_others, lengths)
        forces[i] = LennardJonesForce(pos, nearest_positions)

    return forces


@jit(nopython=True, parallel=False)
def LennardJonesForce(pos1, pos_others, soft_eps=1e-10):
    '''
    Computes the dimensionless force acting on the particle with position 1 due to a Lennard-Jones potential
    caused by particles with dimensionless positions pos_others.

    Parameters
    ----------
    pos1 (ndarray): position of particle on which the force acts. Shape (dim,)
    pos_others (ndarray): positions of particles which cause the force. Shape (n_other_particles, dim).
    soft_eps (float): softening parameter to prevent the force from blowing up for very small distances

    Returns
    -------
    totalForce (ndarray): Cartesian components of the force acting on the particle. Shape (dim,)
    '''

    distances2 = np.zeros(len(pos_others))

    for i in range(len(pos_others)):
        distances2[i] = distanceSquaredFromPosition(pos1, pos_others[i])

    # array calculations for speed
    # these are all 1D arrays of length n_other_particles
    termPauli = -6 / (distances2**3 + soft_eps)
    termWaals = 12 / (distances2**6 + soft_eps)

    # now compute the relative position vector x_i - x_j for each particle j
    # shape is (n_other_particles, dim)
    relativePositions = pos1 - pos_others

    # compute each component of the force due to each individual particle
    # numpy can't automatically broadcast 2D and 1D --> loop over dimensions
    forceTerms = np.zeros(relativePositions.shape)
    for dim in range(relativePositions.shape[-1]):
        forceTerms[:,dim] = relativePositions[:,dim]/(distances2 + soft_eps) * (termPauli + termWaals)

    totalForce = 4 * np.sum(forceTerms, axis=0)

    return totalForce

@jit(nopython=True, parallel=False)
def LennardJonesPotential(pos1, pos_others, soft_eps=1e-10):
    '''
    Computes the Lennard-Jones potential affecting one particle.

    :param pos1: ndarray of shape (dim,)
                Coordinates of the particle under consideration
    :param pos_others: ndarray of shape (n_other_particles, dim)
                Coordinates of the particles causing the potential
    :param soft_eps: float
                Softening parameter. Default 1e-10
    :return: potential: float
                Potential affecting the particle under consideration
    '''

    distances2 = np.zeros(len(pos_others))

    for i in range(len(pos_others)):
        distances2[i] = distanceSquaredFromPosition(pos1, pos_others[i])

    termPauli = -1./(distances2**3 + soft_eps)
    termWaals = 1./(distances2**6 + soft_eps)

    potential_terms = termWaals + termPauli
    potential = 4 * np.sum(potential_terms)

    return potential

class UnitScaler:
    '''
    Scaler used for keeping track of units. Initialised with SI units.

    :param mass: float
                Mass of the atoms in kg. Default 6e6-26 (argon)
    :param length_scale: float
                Interaction length of the potential in m. Default 3.405e-10
    :param energy_scale: float
                Characteristic energy scale of the potential in J. Default 1.654e-21
    '''

    def __init__(self, mass=6.6e-26, length=3.405e-10, energy=1.654e-21):
        self.mass_scale = mass
        self.length_scale = length
        self.energy_scale = energy
        self.k_boltzmann = 1.380649e-23   # J/K

    def toMeters(self, dimless_length):
        '''

        :param dimless_length: float or ndarray
                Dimensionless length
        :return: length: float or ndarray
                Corresponding length in m
        '''

        return self.length_scale * dimless_length

    def toDimlessLength(self, meters):
        '''

        :param meters: float or ndarray
                Length in m
        :return: dimless_length: float or ndarray
                Dimensionless length
        '''

        return meters / self.length_scale

    def toCubicMeters(self, dimless_volume):
        '''

        :param dimless_volume: float or ndarray
                Dimensionless volume
        :return: volume: float or ndarray
                Volume in m^3
        '''

        return self.length_scale**3 * dimless_volume

    def toDimlessVolume(self, cubic_meters):
        '''

        :param cubic_meters: float or ndarray
                Volume in m^3
        :return: dimless_volume: float or ndarray
                Dimensionless volume
        '''

        return cubic_meters / self.length_scale**3

    def toSeconds(self, dimless_time):
        '''

        :param dimless_time: float or ndarray
                Dimensionless time
        :return: time: float or ndarray
                Time in s
        '''

        factor = np.sqrt(self.mass_scale*self.length_scale**2 / self.energy_scale)
        return factor * dimless_time

    def toDimlessTime(self, seconds):
        '''

        :param seconds: float or ndarray
                Time in s
        :return: dimless_time: float or ndarray
                Dimensionless time
        '''

        factor = np.sqrt(self.energy_scale / (self.mass_scale*self.length_scale**2))
        return factor * seconds

    def toMetersPerSecond(self, dimless_vel):
        '''

        :param dimless_vel: float or ndarray
                Dimensionless velocity
        :return: velocity: float or ndarray
                Velocity in m/s
        '''

        factor = np.sqrt(self.energy_scale/self.mass_scale)
        return factor * dimless_vel

    def toDimlessVelocity(self, meters_per_second):
        '''

        :param meters_per_second: float or ndarray
                Velocity in m/s
        :return: dimless_velocity: float or ndarray
                Dimensionless velocity
        '''

        factor = np.sqrt(self.mass_scale / self.energy_scale)
        return factor * meters_per_second

    def toKilogram(self, dimless_mass):
        '''

        :param dimless_mass: float or ndarray
                Dimensionless mass
        :return: mass: float or ndarray
                Mass in kg
        '''
        return self.mass_scale * dimless_mass

    def toDimlessMass(self, kg):
        '''

        :param kg: float or ndarray
                Mass in kg
        :return: dimless_mass: float or ndaaray
                Dimensionless mass
        '''

        return kg / self.mass_scale

    def toJoule(self, dimless_energy):
        '''

        :param dimless_energy: float or ndarray
                Dimensionless energy
        :return: energy: float or ndarray
                Energy in J
        '''

        return self.energy_scale * dimless_energy

    def toDimlessTemperature(self, kelvin):
        '''

        :param kelvin: float or ndarray
                Temperature in K
        :return: dimless_temperature: float or ndarray
                Dimensionless temperature
        '''

        return self.k_boltzmann * kelvin / self.energy_scale

    def toKelvinFromDimlessTemperature(self, dimless_temperature):
        '''

        :param dimless_temperature: float or ndarray
                Dimensionless temperature
        :return: temperature: float or ndarray
                Temperature in K
        '''

        return (self.energy_scale / self.k_boltzmann) * dimless_temperature

    def toKelvinFromDimlessEnergy(self, dimless_energy):
        '''

        :param dimless_energy: float or ndarray
                Dimensionless energy
        :return: temperature: float or ndarray
                Temperature in K
        '''

        return self.toJoule(dimless_energy) / self.k_boltzmann

    def toDimlessEnergy(self, joules):
        '''

        :param joules: float or ndarray
                Energy in J
        :return: dimless_energy: float or ndarray
                Dimensionless energy
        '''

        return joules / self.energy_scale

    def toNewton(self, dimless_force):
        '''

        :param dimless_force: float or ndarray
                Dimensionless force
        :return: force: float or ndarray
                Force in N
        '''

        return self.length_scale**2 / self.energy_scale * dimless_force

    def toDimlessForce(self, newton):
        '''

        :param newton: float or ndarray
                Force in N
        :return: dimless_force: float or ndarray
                Dimensionless force
        '''

        return self.energy_scale / self.length_scale**2 * newton

    def toKilogramPerCubicMeter(self, dimless_density):
        '''

        :param dimless_density: float or ndarray
                Dimensionless density
        :return: density: float or ndarray
                Mass density in kg m^{-3}
        '''

        return (self.mass_scale / self.length_scale**3) * dimless_density

    def toDimlessDensity(self, kg_per_m3):
        '''

        :param kg_per_m3: float or ndarray
                Mass density in kg m^{-3}
        :return: dimless_density: float or ndarray
                Dimensionless density
        '''

        return (self.length_scale**3 / self.mass_scale) * kg_per_m3

    def toJoulePerCubicMeter(self, dimless_pressure):
        '''

        :param dimless_pressure: float or ndarray
                Dimensionless pressure
        :return: pressure: float or ndarray
                Pressure in J m^{-3}
        '''

        return (self.energy_scale / self.length_scale**3) * dimless_pressure

    def toDimlessPressure(self, joule_per_m3):
        '''

        :param joule_per_m3: float or ndarray
                Pressure in J m^{-3}
        :return: dimless_pressure: float or ndarray
                Dimensionless pressure
        '''

        return (self.length_scale**3 / self.energy_scale) * joule_per_m3