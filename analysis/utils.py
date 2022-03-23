import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from framework.particles import Particles
from framework.box import BoxBase

class History:
    def __init__(self, filename):

        # load the saved data set
        datafile = h5py.File(filename, "r")

        dset_pos = datafile["position-history"]
        dset_vel = datafile["velocity-history"]
        dset_energy = datafile["energy-history"]
        dset_times = datafile["times"]

        self.pos = np.copy(dset_pos)
        self.vel = np.copy(dset_vel)
        self.energies = np.copy(dset_energy)

        self.times = np.copy(dset_times)
        self.dim = self.pos.shape[-1]
        self.n_atoms = self.pos.shape[1]
        self.box_edges = dset_pos.attrs["box-edges"]
        self.temperature = dset_pos.attrs["temperature"]
        self.density = dset_pos.attrs["density"]

        datafile.close()


def load_history(historyfile):

    if isinstance(historyfile, History):
        history = historyfile

    elif isinstance(historyfile, str):
        history = History(filename=historyfile)

    else:
        raise TypeError("Given history invalid. Use either the History object or the filename.")

    return history


class ParticlesFromHistory(Particles):
    '''Class for loading a single particle set from a file of simulation results,
    as well as the box used to run the simulations in. By default, the last configuration
    in the particle history is loaded.'''

    def __init__(self, history, time_idx=-1):

        self.history = load_history(history)
        self.time_idx = time_idx

        super().__init__(self.history.n_atoms, self.history.dim)

        self.positions = self.history.pos[time_idx]
        self.velocities = self.history.vel[time_idx]
        self.inputTemp = self.history.temperature

        self.box = BoxBase(self.history.density, self.n_atoms, dim=self.dim)


class SimulationIterations:
    '''Class for loading a series of iterations of a simulation, all stored in datafolder.'''

    def __init__(self, datafolder):

        files_list = os.listdir(datafolder)
        self.histories = []
        self.final_particles = []

        for i, f in enumerate(files_list):
            if f.endswith(".hdf5") or f.endswith(".hdf"):
                history = load_history(datafolder+f)
                self.histories.append(history)

                '''
                particles = Particles(history.n_atoms, history.dim)
                particles.positions = history.pos[-1]
                particles.velocities = history.vel[-1]

                '''
                particles = ParticlesFromHistory(history)
                self.final_particles.append(particles)


                self.final_particles.append(particles)

        self.datafolder = datafolder
        self.box = self.final_particles[0].box


class PlotPreferences:
    def __init__(self, usetex=False, markersize=3, figsize=(7,5), dpi=240,\
                 marker="o"):

        if usetex:
            plt.rcParams["text.usetex"] = True

        plt.rcParams["font.family"] = "serif"

        self.markersize = markersize
        self.marker = marker
        self.figsize = figsize
        self.dpi = dpi


class RepeatedSimsBase:
    '''Base class for inferring various useful global quantities for
    a number of random realisations of the same initial conditions.'''

    def __init__(self, particles, box_lengths):

        if isinstance(particles, list):
            self.particles = particles
        else:
            self.particles = [particles]

        self.box_lengths = box_lengths
        self.n_iterations = len(self.particles)
        self.n_atoms = self.particles[0].n_atoms
        self.dim = self.particles[0].dim

        if isinstance(self.box_lengths, np.ndarray) and len(self.box_lengths.shape) == 1:
            self.volume = self.box_lengths[0] ** self.dim

        elif isinstance(self.box_lengths, float):
            self.volume = self.box_lengths ** self.dim

        else:
            raise ValueError("Invalid box_lengths given. Give either a 1D array or a float.")

        self.temperature = self.particles[0].temperature
        self.density = self.n_atoms / self.volume


class VaryingInitialConditionsSims(RepeatedSimsBase):

    def __init__(self, particles, box_lengths):
        super().__init__(particles, box_lengths)

        self.n_atoms = np.zeros(len(particles))
        self.temperature = np.zeros(len(particles))
        self.density = np.zeros(len(particles))

        for i, set in enumerate(particles):
            self.n_atoms[i] = set.n_atoms
            self.temperature[i] = set.temperature
            self.density[i] = set.n_atoms / self.volume