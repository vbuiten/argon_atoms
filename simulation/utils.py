'''Module containing utility functions for doing argon simulations.'''

import numpy as np

def distanceFromPosition(pos1, pos2):
    '''Calculates the distance between 1D arrays pos1 and pos2.'''

    diffVector = pos2 - pos1
    dist = np.sqrt(np.dot(diffVector, diffVector))

    return dist


# need a function for evaluating the gradient of the Lennard-Jones potential
# also need some class for storing constants and physical units

def LennardJonesForce(pos1, pos_others, eps=119.8, sigma=3.405, soft_eps=0.00001):
    '''
    Computes the force acting on the particle with position 1 due to a Lennard-Jones potential
    caused by particles with positions pos_others.

    Parameters
    ----------
    pos1 (ndarray): position of particle on which the force acts.
    pos_others (ndarray): positions of particles which cause the force. Shape (n_particles, dim).
    eps: value of dielectric permittivity.
    sigma: value of length scale sigma.

    Returns
    -------
    totalForce (ndarray): Cartesian components of the force acting on the particle
    '''

    distances = np.zeros(len(pos_others))

    for i in range(len(pos_others)):
        distances[i] = distanceFromPosition(pos1, pos_others[i])

    # array calculations for speed
    # these are all 1D arrays of length n_other_particles
    lengthFractions = sigma/distances
    termPauli = -6 * (lengthFractions)**6
    termWaals = 12 * (lengthFractions)**12

    # now compute the relative position vector x_i - x_j for each particle j
    # shape is (n_other_particles, dim)
    relativePositions = pos1 - pos_others

    # compute each component of the force due to each individual particle
    # numpy can't automatically broadcast 2D and 1D --> loop over dimensions
    forceTerms = np.zeros(relativePositions.shape)
    for dim in range(relativePositions.shape[-1]):
        forceTerms[:,dim] = relativePositions[:,dim]/(distances**2 + soft_eps) * (termPauli + termWaals)

    totalForce = 4*eps * np.sum(forceTerms, axis=0)

    return totalForce

