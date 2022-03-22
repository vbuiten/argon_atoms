import h5py
import numpy as np
import matplotlib.pyplot as plt

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


class ParticlesFromHistory:
    def __init__(self, history):

        self.history = load_history(history)




class PlotPreferences:
    def __init__(self, usetex=False, markersize=3, figsize=(7,5), dpi=240,\
                 marker="o"):

        if usetex:
            plt.rcParams["text.usetex"] = True

        else:
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