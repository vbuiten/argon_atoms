'''Module containing the simulation worker class.'''

import numpy as np
from simulation.utils import LennardJonesForce

class NBodyWorker:
    def __init__(self, bodies, box, timestep=0.1):

        # check if box and bodies have the same dimensions
        if box.dim != bodies.dim:
            raise ValueError ("Dimensions of bodies and box do not match!")

        self.bodies = bodies
        self.box = box
        self.time = 0
        self.timestep = timestep


    def evolve(self, t_end):

        times = np.arange(self.time, self.time+t_end, self.timestep)

        for time in times:
            newpos = self.bodies.positions + self.bodies.velocities * self.timestep
            forces = np.zeros(self.bodies.velocities.shape)

            for i in range(len(self.bodies)):
                pos = self.bodies.positions[i]
                pos_others = np.concatenate((self.bodies.positions[:i], self.bodies.positions[i+1:]))
                forces[i] = LennardJonesForce(pos, pos_others)

            newvel = self.bodies.velocities + forces * self.timestep / self.bodies.mass

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

        print ("Simulation finished.")