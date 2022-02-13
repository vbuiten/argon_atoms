'''Module containing utility functions for doing argon simulations.'''

import numpy as np

def distanceFromPosition(pos1, pos2):
    '''Calculates the distance between 1D arrays pos1 and pos2.'''

    diffVector = pos2 - pos1
    dist = np.sqrt(np.dot(diffVector, diffVector))

    return dist


# need a function for evaluating the gradient of the Lennard-Jones potential
# also need some class for storing constants and physical units