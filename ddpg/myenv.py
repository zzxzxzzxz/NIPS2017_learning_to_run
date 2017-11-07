from osim.env import RunEnv
from feature_generator import Observation

def dist(x0, y0, x1, y1):
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


class MyRunEnv(RunEnv):

    def step(self, action):
        observation, reward, done, info = super(MyRunEnv, self).step(action)
        obs = Observation(observation)

        nz = (obs.head_y - obs.pelvis_y) / dist(obs.head_x, obs.head_y, obs.pelvis_x, obs.pelvis_y)
        reward += 0.0005 * nz

        if self.istep >= 70:
            if obs.left_knee_r > 0.015:
                reward -= 0.01
            if obs.right_knee_r > 0.015:
                reward -= 0.01

        return observation, reward, done, info
