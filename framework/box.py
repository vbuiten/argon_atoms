import numpy as np

class BoxBase:
    '''Class setting up the box in which the particles live.'''

    def __init__(self, lengths):

        if isinstance(lengths, int):
            self.lengths = (lengths,)

        else:
            self.lengths = lengths

        self.dim = len(self.lengths)

        edges = np.zeros((self.dim,2))
        edges[:,1] = lengths
        self.edges = edges