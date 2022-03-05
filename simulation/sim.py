'''Module containing the simulation worker class.'''

import numpy as np
from simulation.utils import LennardJonesForce
import h5py

class NBodyWorker:
    def __init__(self, bodies, box, timestep=0.1):

        # check if box and bodies have the same dimensions
        if box.dim != bodies.dim:
            raise ValueError ("Dimensions of bodies and box do not match!")

        self.bodies = bodies
        self.box = box
        self.time = 0
        self.timestep = timestep


    def saveToFile(self, savefile, times):

        file = h5py.File(savefile, "w")
        pos_dataset = file.create_dataset("position-history", data=self.pos_history)
        vel_dataset = file.create_dataset("velocity-history", data=self.vel_history)

        pos_dataset.attrs["times"] = times
        vel_dataset.attrs["times"] = times

        print (len(pos_dataset))

        file.close()

        print ("File created.")


    def evolve(self, t_end, savefile=None, timestep_external=1.):

        times = np.arange(self.time, self.time+t_end, self.timestep)
        times_external = []

        pos_history = np.zeros((len(times), len(self.bodies), self.box.dim))
        vel_history = np.zeros((len(times), len(self.bodies), self.box.dim))

        for idx, time in enumerate(times):

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
                pos_others = (pos - pos_others + length/2) % length - length/2
                #pos_others = pos_others - length * (pos_diff/length).astype(int)
                forces[i] = LennardJonesForce(pos, pos_others)

            # now update the positions
            posadd = self.bodies.velocities * self.timestep

            edges_cast = np.broadcast_to(self.box.edges, np.concatenate((posadd.shape, (2,))))
            newpos = edges_cast[:,:,0] + (self.bodies.positions + posadd + 2*(edges_cast[:,:,1]-edges_cast[:,:,0])) % (edges_cast[:,:,1] - edges_cast[:,:,0])

            # and update the velocities
            newvel = self.bodies.velocities + forces * self.timestep / self.bodies.mass

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

            if time/timestep_external % 1 == 0:
                times_external.append(time)
                print ("Time:", time)

            pos_history[idx] = self.bodies.positions
            vel_history[idx] = self.bodies.velocities

        self.time = times[-1]
        times_external = np.array(times_external)

        if savefile is not None:
            # create a file
            self.pos_history = pos_history
            self.vel_history = vel_history

            self.saveToFile(savefile, times)

        print ("Simulation finished.")