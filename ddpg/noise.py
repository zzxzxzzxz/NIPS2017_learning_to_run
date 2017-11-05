import numpy as np

class one_fsq_noise(object):
    def __init__(self, skip=1):
        self.buffer = np.array([0.])
        self.state = np.random.RandomState()
        self.skip = skip
        self.count = 0

    def one(self,size,noise_level=1.):
        if self.buffer.shape != size:
            self.buffer = np.zeros(size, dtype='float32')

        if self.count == 0:
            g = self.state.normal(loc=0., scale=noise_level, size=size)
            self.buffer += g
            self.buffer *= .9
        self.count = (self.count + 1) % self.skip

        return self.buffer.copy()

    def ask(self):
        return self.buffer.copy()

# 1/f^2 noise: http://hal.in2p3.fr/in2p3-00024797/document
