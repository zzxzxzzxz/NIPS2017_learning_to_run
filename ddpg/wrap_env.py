from multiprocessing import Process, Pipe

# FAST ENV

# this is a environment wrapper. it wraps the RunEnv and provide interface similar to it. The wrapper do a lot of pre and post processing (to make the RunEnv more trainable), so we don't have to do them in the main program.

#from observation_processor import generate_observation as go
from feature_generator2 import FeatureGenerator
import numpy as np

class fastenv:
    def __init__(self,e,skipcount):
        self.e = e
        self.stepcount = 0
        self.skipcount = skipcount
        self.fg = None

    def obg(self,plain_obs):
        processed_observation = self.fg.gen(plain_obs)
        return np.array(processed_observation)

    def step(self,action):
        action = [float(action[i]) for i in range(len(action))]

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met',action)
                raise RuntimeError('this is bullshit')

        sr = 0
        for j in range(self.skipcount):
            self.stepcount+=1
            oo,r,d,i = self.e.step(action)
            o = self.obg(oo)
            sr += r

            if d == True:
                break

        return o, sr, d, i

    def reset(self):
        self.stepcount=0
        self.fg = FeatureGenerator()

        oo = self.e.reset()
        self.lastx = oo[1]
        o = self.obg(oo)
        return o
