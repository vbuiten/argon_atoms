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
            posadd = self.bodies.velocities * self.timestep
            edges_cast = np.broadcast_to(self.box.edges, np.concatenate((posadd.shape, (2,))))
            newpos = edges_cast[:,:,0] + (self.bodies.positions + posadd) % (edges_cast[:,:,1] - edges_cast[:,:,0])

            #for i in range(len(newpos)):
            #    for j in range(self.box.dim):
            #        if (newpos[i,j] < self.box.edges[j,0]) | (newpos[i,j] > self.box.edges[j,1]):
            #            newpos[i,j] = self.box.edges[j,0] + (self.bodies.positions[i,j] + posadd[i,j]) % (self.box.edges[j,1] - self.box.edges[j,0])

            # if the new position crosses the box edges, the particle comes out on the other side
            #print ("Shape of inbox:", inbox.shape)

            # broadcast the edges
            #edges_cast = np.broadcast_to(self.box.edges, np.concatenate((newpos.shape, (2,))))
            #inbox = (newpos > edges_cast[:,:,0]) & (newpos < edges_cast[:,:,1])
            #print ("Shape of edges_cast:", edges_cast.shape)
            #print ("Shape of edges_cast[~inbox]:", edges_cast[~inbox].shape)
            #newpos[~inbox] = edges_cast[~inbox][:,0] + (posadd[~inbox] - (edges_cast[~inbox][:,1] - self.bodies.positions[~inbox]))

            forces = np.zeros(self.bodies.velocities.shape)

            for i in range(len(self.bodies)):
                pos = self.bodies.positions[i]
                pos_others = np.concatenate((self.bodies.positions[:i], self.bodies.positions[i+1:]))
                forces[i] = LennardJonesForce(pos, pos_others, soft_eps=0)

            newvel = self.bodies.velocities + forces * self.timestep / self.bodies.mass

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

            print ("Time:", time)

        self.time = times[-1]
        #print ("Forces:", forces)

        print ("Simulation finished.")