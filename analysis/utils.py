import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from framework.particles import Particles
from framework.box import BoxBase

class History:
    '''
    Class for loading simulation data.

    :param filename: str
            Absolute or relative path of the hdf5 file containing the simulation data
    '''

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
    '''
    Utility function for loading simulation data from either a file or a pre-loaded History instance.

    :param historyfile: str or History instance
            File or object containing the simulation data
    :return: history: History instance
            History object containing the simulation data
    '''

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
    in the particle history is loaded.

    :param history: str of History instance
            File or object containing simulation data
    :param time_idx: int
            Index of the saved time step for which to load the particles. Default -1 (i.e. the final configuration)
    '''

    def __init__(self, history, time_idx=-1):

        self.history = load_history(history)
        self.time_idx = time_idx

        super().__init__(self.history.n_atoms, self.history.dim)

        self.positions = self.history.pos[time_idx]
        self.velocities = self.history.vel[time_idx]
        self.inputTemp = self.history.temperature

        self.box = BoxBase(self.history.density, self.n_atoms, dim=self.dim)


class SimulationIterations:
    '''Class for loading a series of iterations of a simulation, all stored in datafolder.

    :param datafolder: str
            Folder containing all the simulation data for the initial conditions of interest
    :param samebox: bool
            Whether or not all simulations have the same box size. Default is True
    '''

    def __init__(self, datafolder, samebox=True):

        files_list = os.listdir(datafolder)
        self.histories = []
        self.final_particles = []
        boxes = []

        for i, f in enumerate(files_list):
            if f.endswith(".hdf5") or f.endswith(".hdf"):
                history = load_history(datafolder+f)
                self.histories.append(history)

                particles = ParticlesFromHistory(history)
                self.final_particles.append(particles)

                boxes.append(particles.box)

        self.datafolder = datafolder

        if samebox:
            self.box = boxes[0]

        else:
            self.box = boxes


class PlotPreferences:
    '''
    Class for holding several layout preferences for plots.

    :param usetex: bool
            If True, uses Latex compilation
    :param markersize: float
            Sets the marker size
    :param figsize: tuple of length 2
            Sets the figure size
    :param dpi: int
            Sets the dpi of the figure
    :param marker: str
            Sets the type of marker to use in plots
    '''

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
    a number of random realisations of the same initial conditions.

    :param particles: framework.particles.Particles instance or list thereof
            Particle system(s) to analyse
    :param box_lengths: float or ndarray of shape (n_sims,) or (dim,)
            Linear size of the box. The first index is assumed to be the length in all simulations and dimensions.

    '''

    def __init__(self, particles, box_lengths):

        if isinstance(particles, list):
            self.particles = particles
        else:
            self.particles = [particles]

        self.box_lengths = box_lengths
        self.n_iterations = len(self.particles)
        self.n_atoms = self.particles[0].n_atoms
        self.dim = self.particles[0].dim

        if isinstance(self.box_lengths, np.ndarray):
            if len(self.box_lengths.shape) == 1:
                self.volume = self.box_lengths[0] ** self.dim

        elif isinstance(self.box_lengths, float):
            self.volume = self.box_lengths ** self.dim

        else:
            raise ValueError("Invalid box_lengths given. Give either a 1D array or a float.")

        self.temperature = self.particles[0].temperature
        self.density = self.n_atoms / self.volume


class VaryingInitialConditionsSims:
    '''
    Base class for analysing simulations of varying initial conditions.

    :param particles: framework.particles.Particles instance or list thereof
            Particle system(s) to analyse
    :param boxes: list of framework.box.BoxBase instances
            BoxBase instances corresponding the particle systems.
    '''

    def __init__(self, particles, boxes):

        if isinstance(particles, list):
            self.particles = particles
        else:
            self.particles = [particles]

        self.n_iterations = len(self.particles)
        self.dim = self.particles[0].dim

        self.n_atoms = np.zeros(len(particles))
        self.temperature = np.zeros(len(particles))
        self.density = np.zeros(len(particles))

        box_lengths = []
        for i, el in enumerate(boxes):
            box_lengths.append(el.lengths)

        self.box_lengths = np.array(box_lengths)
        self.volume = self.box_lengths[:,0] ** self.dim

        for i, set in enumerate(particles):
            self.n_atoms[i] = set.n_atoms
            self.temperature[i] = set.temperature
            self.density[i] = set.n_atoms / self.volume[i]