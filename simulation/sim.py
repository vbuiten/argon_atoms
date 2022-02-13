'''Module containing the simulation worker class.'''

import numpy as np

class NBodyWorker:
    def __init__(self, bodies, box, timestep=0.1):

        # check if box and bodies have the same dimensions
        if box.dim != bodies.dim:
            raise ValueError ("Dimensions of bodies and box do not match!")

        self.bodies = bodies
        self.box = box
        self.time = 0
        self.timestep = timestep

