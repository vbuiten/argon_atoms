import h5py
import numpy as np

class History:
    def __init__(self, filename):

        # load the saved data set
        datafile = h5py.File(filename, "r")

        dset_pos = datafile["position-history"]
        dset_vel = datafile["velocity-history"]

        self.pos = np.copy(dset_pos)
        self.vel = np.copy(dset_vel)

        self.times = np.copy(dset_pos.attrs["times"])
        self.dim = self.pos.shape[-1]
        self.n_atoms = self.pos.shape[1]

        datafile.close()