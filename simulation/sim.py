'''Module containing the simulation worker class.'''

import numpy as np
from simulation.utils import LennardJonesForce, posInBox, minimumImageForces
import h5py

class NBodyWorker:
    def __init__(self, bodies, box, timestep=0.1, method="Verlet", minimage=True):

        # check if box and bodies have the same dimensions
        if box.dim != bodies.dim:
            raise ValueError ("Dimensions of bodies and box do not match!")

        self.bodies = bodies
        self.box = box
        self.time = 0
        self.timestep = timestep
        self.method = method
        self.minimage = minimage


    def saveToFile(self, savefile, times):

        file = h5py.File(savefile, "w")
        pos_dataset = file.create_dataset("position-history", data=self.pos_history)
        vel_dataset = file.create_dataset("velocity-history", data=self.vel_history)
        E_kin_dataset = file.create_dataset("energy-history", data=self.energy_history)

        pos_dataset.attrs["times"] = times
        vel_dataset.attrs["times"] = times

        pos_dataset.attrs["box-edges"] = self.box.edges
        vel_dataset.attrs["box-edges"] = self.box.edges
        E_kin_dataset.attrs["box-edges"] = self.box.edges

        print (len(pos_dataset))

        file.close()

        print ("File created.")


    def equilibriate(self, iterations=50, iteration_time=100., threshold=1.e-2):

        target_kinetic_energy = (self.bodies.dim/2) * (self.bodies.n_atoms - 1) * self.bodies.dimlessTemp

        times = np.arange(0., iteration_time, self.timestep)
        length = self.box.lengths[0]

        # save the energy fractions for each iteration
        energy_fractions = []

        # use Euler integration backwards for the positions at t = -1
        if self.method == "Verlet":
            old_pos = posInBox(self.bodies.positions - self.timestep * self.bodies.velocities, self.box.lengths)

        fractional_deviation = np.inf

        #while np.abs(fractional_deviation) > threshold:
        for i in range(iterations):

            # run the simulation for a time iteration_time

            for idx, time in enumerate(times):

                if self.minimage:
                    # first compute the force acting on each particle
                    forces = minimumImageForces(self.bodies.positions, self.box.lengths)

                else:
                    forces = np.zeros(self.bodies.positions.shape)
                    for i in range(len(forces)):
                        pos = self.bodies.positions[i]
                        pos_others = np.concatenate((self.bodies.positions[:i], self.bodies.positions[i + 1:]))
                        forces[i] = LennardJonesForce(pos, pos_others)

                # update positions and velocities
                # use the user-specified algorithm
                if self.method == "Euler":
                    newpos = self.bodies.positions + self.timestep * self.bodies.velocities
                    newpos = posInBox(newpos, self.box.lengths)

                    newvel = self.bodies.velocities + self.timestep * forces

                elif self.method == "Verlet":
                    newpos = 2 * self.bodies.positions - old_pos + self.timestep ** 2 * forces
                    newpos = posInBox(newpos, self.box.lengths)

                    newvel = (newpos - old_pos) / (2 * self.timestep)

                    old_pos = self.bodies.positions

                else:
                    raise ValueError("Invalid integration method given. Use 'Euler' or 'Verlet'.")

                self.bodies.positions = newpos
                self.bodies.velocities = newvel

            #self.evolve(self.time+iteration_time)
            #self.evolve(iteration_time)

            # measure the new kinetic energy after evolving
            real_kinetic_energy = self.bodies.kineticEnergy()
            energy_fraction = target_kinetic_energy / real_kinetic_energy
            fractional_deviation = energy_fraction - 1

            energy_fractions.append(energy_fraction)
            print("target E_kin / real E_kin =", energy_fraction)
            #print("target E_kin / real E_kin - 1 =", fractional_deviation)

            if np.abs(fractional_deviation) > threshold:
                vel_scale_factor = np.sqrt(energy_fraction)

                #print ("Scale factor:", vel_scale_factor)

                # rescale the velocities
                newvel = vel_scale_factor * self.bodies.velocities
                self.bodies.velocities = newvel

                print ("Rescaled velocities.")


        print ("Equilibriation complete.")

        #self.time = 0
        return energy_fractions


    def evolve(self, t_end, savefile=None, timestep_external=1.):

        times = np.arange(self.time, self.time+t_end, self.timestep)
        times_external = []

        #pos_history = np.zeros((len(times_external), len(self.bodies), self.box.dim))
        #vel_history = np.zeros((len(times_external), len(self.bodies), self.box.dim))
        pos_history = []
        vel_history = []
        kinetic_energy = []
        potential_energy = []

        length = self.box.length

        if self.method == "Verlet":
            # compute the "previous set" of positions (backward Euler)
            pos_subtract = self.timestep * self.bodies.velocities
            old_pos = posInBox(self.bodies.positions - pos_subtract, self.box.lengths)

        for idx, time in enumerate(times):

            # save the current state of the system
            if (time - self.time)/timestep_external % 1 == 0:
                times_external.append(time)
                print ("Time:", time)
                #print ("Forces:", forces)
                pos_history.append(self.bodies.positions)
                vel_history.append(self.bodies.velocities)
                kinetic_energy.append(self.bodies.kineticEnergy())

                # computationally expensive potential energy calculation
                potential_energy.append(self.bodies.potentialEnergy(length))

            # now evolve the system to the next step

            if self.minimage:
                # first compute the force acting on each particle
                forces = minimumImageForces(self.bodies.positions, self.box.lengths)

            else:
                # compute forces without minimum image convention
                forces = np.zeros(self.bodies.positions.shape)
                for i in range(len(forces)):
                    pos = self.bodies.positions[i]
                    pos_others = np.concatenate((self.bodies.positions[:i], self.bodies.positions[i+1:]))
                    forces[i] = LennardJonesForce(pos, pos_others)

            # update positions and velocities
            # use the user-specified algorithm
            if self.method == "Euler":
                newpos = self.bodies.positions + self.timestep * self.bodies.velocities
                newpos = posInBox(newpos, self.box.lengths)

                newvel = self.bodies.velocities + self.timestep * forces

            elif self.method == "Verlet":
                newpos = 2*self.bodies.positions - old_pos + self.timestep**2 * forces
                newpos = posInBox(newpos, self.box.lengths)

                newvel = (newpos - old_pos) / (2*self.timestep)

                # save the current positions as "old positions" for the next iteration
                old_pos = self.bodies.positions

            else:
                raise ValueError("Invalid integration method given. Use 'Euler' or 'Verlet'.")

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

        # total energy
        total_energy = np.array(kinetic_energy) + np.array(potential_energy)

        pos_history = np.array(pos_history)
        vel_history = np.array(vel_history)
        energy_history = np.array([times_external, kinetic_energy, potential_energy, total_energy]).T
            #pos_history[idx] = self.bodies.positions
            #vel_history[idx] = self.bodies.velocities

        self.time = times[-1]
        times_external = np.array(times_external)

        if savefile is not None:
            # create a file
            self.pos_history = pos_history
            self.vel_history = vel_history
            self.energy_history = energy_history

            self.saveToFile(savefile, times_external)

        print ("Simulation finished.")
        #print ("Internal times:", times)
        #print ("External times:", times_external)