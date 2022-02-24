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

            # first compute the force acting on each particle
            forces = np.zeros(self.bodies.velocities.shape)

            for i in range(len(self.bodies)):
                pos = self.bodies.positions[i]
                pos_others = np.concatenate((self.bodies.positions[:i], self.bodies.positions[i + 1:]))

                # implement the minimum image convention
                # for simplicity we use a rectangular/cubic box
                # rather than a circle/sphere
                pos_diff = pos_others - pos
                length = self.box.lengths[0]
                pos_others = pos_others - length * (pos_diff/length).astype(int)
                forces[i] = LennardJonesForce(pos, pos_others, soft_eps=0)

            # now update the positions
            posadd = self.bodies.velocities * self.timestep

            edges_cast = np.broadcast_to(self.box.edges, np.concatenate((posadd.shape, (2,))))
            newpos = edges_cast[:,:,0] + (self.bodies.positions + posadd + 2*(edges_cast[:,:,1]-edges_cast[:,:,0])) % (edges_cast[:,:,1] - edges_cast[:,:,0])

            # and update the velocities
            newvel = self.bodies.velocities + forces * self.timestep / self.bodies.mass

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

            if time/self.timestep % 100 == 0:
                print ("Time:", time)

        self.time = times[-1]

        print ("Simulation finished.")