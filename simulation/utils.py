'''Module containing utility functions for doing argon simulations.'''

import numpy as np

def distanceFromPosition(pos1, pos2):
    '''Calculates the distance between 1D arrays pos1 and pos2.'''

    diffVector = pos2 - pos1
    dist = np.sqrt(np.dot(diffVector, diffVector))

    return dist


def periodicCopies(positions, length):
    '''Create periodic copies for each particle specified by positions.'''

    n_particles = len(positions)
    dim = positions.shape[-1]
    copies = np.zeros((n_particles, 9, dim))
    print ("copies.shape:", copies.shape)

    for i in range(n_particles):
        for j in (copies.shape[1],):
            for k in (copies.shape[-1],):
                copies[i,j,k] = positions[i,k] - length + 0.5*j*length

    return copies


def posInBox(pos, edges):
    '''Simple function for shifting a particle position inside the box.'''

    lengths = edges[:,1] - edges[:,0]
    pos = edges[:,0] + (pos + 2*lengths) % lengths
    return pos


def minimumImageForces(positions, edges):

    forces = np.zeros(positions.shape)
    lengths = edges[:,1] - edges[:,0]

    for i in range(len(forces)):
        pos = positions[i]
        pos_others = np.concatenate((positions[:i], positions[i+1:]))

        pos_diff = pos_others - pos
        nearest_positions = pos_others - lengths * np.rint(pos_diff/lengths)
        forces[i] = LennardJonesForce(pos, nearest_positions)

    return forces


def LennardJonesForce(pos1, pos_others, soft_eps=0.0001):
    '''
    Computes the dimensionless force acting on the particle with position 1 due to a Lennard-Jones potential
    caused by particles with dimensionless positions pos_others.

    Parameters
    ----------
    pos1 (ndarray): position of particle on which the force acts.
    pos_others (ndarray): positions of particles which cause the force. Shape (n_particles, dim).

    Returns
    -------
    totalForce (ndarray): Cartesian components of the force acting on the particle
    '''

    distances = np.zeros(len(pos_others))

    for i in range(len(pos_others)):
        distances[i] = distanceFromPosition(pos1, pos_others[i])

    # array calculations for speed
    # these are all 1D arrays of length n_other_particles
    termPauli = -6 * (distances + soft_eps)**-6
    termWaals = 12 * (distances + soft_eps)**-12

    # now compute the relative position vector x_i - x_j for each particle j
    # shape is (n_other_particles, dim)
    relativePositions = pos1 - pos_others

    # compute each component of the force due to each individual particle
    # numpy can't automatically broadcast 2D and 1D --> loop over dimensions
    forceTerms = np.zeros(relativePositions.shape)
    for dim in range(relativePositions.shape[-1]):
        forceTerms[:,dim] = relativePositions[:,dim]/(distances**2 + soft_eps) * (termPauli + termWaals)

    totalForce = 4 * np.sum(forceTerms, axis=0)

    return totalForce


def LennardJonesPotential(pos1, pos_others, soft_eps=0.0001):

    distances = np.zeros(len(pos_others))

    for i in range(len(pos_others)):
        distances[i] = distanceFromPosition(pos1, pos_others[i])

    termPauli = -(distances + soft_eps)**-6
    termWaals = (distances + soft_eps)**-12

    potential_terms = 4 * (termWaals + termPauli)
    potential = np.sum(potential_terms)

    return potential

class UnitScaler:
    '''Scaler used for keeping track of units. Initialised with SI units.'''

    def __init__(self, mass=6.6e-26, length=3.405e-10, energy=1.654e-21):
        self.mass_scale = mass
        self.length_scale = length
        self.energy_scale = energy
        self.k_boltzmann = 1.380649e-23   # J/K

    def toMeters(self, dimless_length):
        return self.length_scale * dimless_length

    def toDimlessLength(self, meters):
        return meters / self.length_scale

    def toCubicMeters(self, dimless_volume):
        return self.length_scale**3 * dimless_volume

    def toDimlessVolume(self, cubic_meters):
        return cubic_meters / self.length_scale**3

    def toSeconds(self, dimless_time):
        factor = np.sqrt(self.mass_scale*self.length_scale**2 / self.energy_scale)
        return factor * dimless_time

    def toDimlessTime(self, seconds):
        factor = np.sqrt(self.energy_scale / (self.mass_scale*self.length_scale**2))
        return factor * seconds

    def toMetersPerSecond(self, dimless_vel):
        factor = np.sqrt(self.energy_scale/self.mass_scale)
        return factor * dimless_vel

    def toDimlessVelocity(self, meters_per_second):
        factor = np.sqrt(self.mass_scale / self.energy_scale)
        return factor * meters_per_second

    def toKilogram(self, dimless_mass):
        return self.mass_scale * dimless_mass

    def toDimlessMass(self, kg):
        return kg / self.mass_scale

    def toJoule(self, dimless_energy):
        return self.energy_scale * dimless_energy

    def toDimlessTemperature(self, kelvin):
        return self.k_boltzmann * kelvin / self.energy_scale

    def toKelvin(self, dimless_energy):
        return self.toJoule(dimless_energy) / self.k_boltzmann

    def toDimlessEnergy(self, joules):
        return joules / self.energy_scale

    def toNewton(self, dimless_force):
        return self.length_scale**2 / self.energy_scale * dimless_force

    def toDimlessForce(self, newton):
        return self.energy_scale / self.length_scale**2 * newton

    def toKilogramPerCubicMeter(self, dimless_density):
        return (self.mass_scale / self.length_scale**3) * dimless_density

    def toDimlessDensity(self, kg_per_m3):
        return (self.length_scale**3 / self.mass_scale) * kg_per_m3