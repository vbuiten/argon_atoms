import h5py
import numpy as np

class History:
    def __init__(self, filename):

        # load the saved data set
        datafile = h5py.File(filename, "r")

        self.pos_history = datafile["position-history"]
        self.vel_history = datafile["velocity-history"]

        datafile.close()

        #self.timesteps = len(self.pos_history)