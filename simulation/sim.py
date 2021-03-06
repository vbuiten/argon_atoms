'''Module containing the simulation worker class.'''

import numpy as np
from simulation.utils import LennardJonesForce, posInBox, minimumImageForces
import h5py
from numba import jit

class Simulator:
    '''
    Class for equilibrating and running the simulation, and saving the results.

    :param: bodies: framework.particles.Particles instance
    :param box: framework.box.BoxBase instance
    :param timestep=0.001: dimensionless time step to use.
    :param method="Verlet": integration method to use
    :param minimage=True: whether to use the minimum image convention
    '''

    def __init__(self, bodies, box, timestep=0.001, method="Verlet", minimage=True):

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
        '''
        Save the simulation data to a file.

        :param savefile: filename to save data to
        :param times: time stamps to save
        '''

        file = h5py.File(savefile, "w")
        pos_dataset = file.create_dataset("position-history", data=self.pos_history)
        vel_dataset = file.create_dataset("velocity-history", data=self.vel_history)
        energy_dataset = file.create_dataset("energy-history", data=self.energy_history)

        times_dataset = file.create_dataset("times", data=times)

        pos_dataset.attrs["temperature"] = self.bodies.inputTemp
        vel_dataset.attrs["temperature"] = self.bodies.inputTemp

        pos_dataset.attrs["density"] = self.box.density
        vel_dataset.attrs["density"] = self.box.density

        pos_dataset.attrs["box-edges"] = self.box.edges
        vel_dataset.attrs["box-edges"] = self.box.edges
        energy_dataset.attrs["box-edges"] = self.box.edges

        print (len(pos_dataset))

        file.close()

        print ("File created.")


    def equilibrate(self, iterations=15, iteration_time=10., threshold=1.e-2):
        '''
        Equilibrate the system.

        :param iterations: maximum number of iterations
        :param iteration_time: dimensionless time for one equilibration step
        :param threshold: maximum deviation from the desired kinetic energy; sets the stopping condition

        :return energy_fractions: target kinetic energy / actual kinetic energy for all equilibration steps.
        '''

        target_kinetic_energy = (self.bodies.dim/2) * (self.bodies.n_atoms - 1) * self.bodies.inputTemp

        # save the energy fractions for each iteration
        energy_fractions = []

        for i in range(iterations):

            # run the simulation for a time iteration_time

            self.evolve(iteration_time)

            # measure the new kinetic energy after evolving
            real_kinetic_energy = self.bodies.kineticEnergy()
            energy_fraction = target_kinetic_energy / real_kinetic_energy
            fractional_deviation = energy_fraction - 1

            energy_fractions.append(energy_fraction)
            print("target E_kin / real E_kin =", energy_fraction)

            if np.abs(fractional_deviation) > threshold:

                vel_scale_factor = np.sqrt(energy_fraction)

                # rescale the velocities
                newvel = vel_scale_factor * self.bodies.velocities

                self.bodies.velocities = newvel

                print ("Rescaled velocities.")

            else:
                break


        print ("Equilibriation complete.")

        self.time = 0
        return energy_fractions


    def evolve(self, t_end, savefile=None, timestep_external=1.):
        '''
        Evolves the system.

        :param t_end: time to run the system (dimensionless units)
        :param savefile: file to save data to. If None, saves no data.
        :param timestep_external: time step to use for saving data.
        '''

        times = np.arange(self.time, self.time+t_end, self.timestep)

        if savefile is not None:
            times_external = []
            pos_history = []
            vel_history = []
            kinetic_energy = []
            potential_energy = []

        length = self.box.length

        for idx, time in enumerate(times):

            if savefile is not None:
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

                if idx == 0:
                    # compute the "previous set" of positions (backward Euler)
                    # these have to ignore the "stay in box" condition!
                    old_pos = self.bodies.positions - self.timestep * self.bodies.velocities
                    # old_pos = self.bodies.positions + pos_subtract
                    current_pos = np.copy(self.bodies.positions)

                newpos = 2*current_pos - old_pos + self.timestep**2 * forces

                # compute velocity before shifting positions to stay inside the box
                newvel = (newpos - old_pos) / (2 * self.timestep)

                # save the current positions as "old positions" for the next iteration
                old_pos = np.copy(current_pos)
                current_pos = np.copy(newpos)

                newpos = posInBox(newpos, self.box.lengths)

            else:
                raise ValueError("Invalid integration method given. Use 'Euler' or 'Verlet'.")

            self.bodies.positions = newpos
            self.bodies.velocities = newvel

        self.time = times[-1]

        if savefile is not None:

            times_external = np.array(times_external)
            total_energy = np.array(kinetic_energy) + np.array(potential_energy)

            pos_history = np.array(pos_history)
            vel_history = np.array(vel_history)
            energy_history = np.array([times_external, kinetic_energy, potential_energy, total_energy]).T

            # create a file
            self.pos_history = pos_history
            self.vel_history = vel_history
            self.energy_history = energy_history

            self.saveToFile(savefile, times_external)

        print ("Simulation finished.")