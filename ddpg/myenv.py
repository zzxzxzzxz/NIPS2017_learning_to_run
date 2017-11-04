from osim.env import RunEnv
from feature_generator import Observation

class MyRunEnv(RunEnv):

    def step(self, action):
        observation, reward, done, info = super(MyRunEnv, self).step(action)
        obs = Observation(observation)

        if self.istep >= 70:
            if obs.left_knee_r > 0.015:
                reward -= 0.01
            if obs.right_knee_r > 0.015:
                reward -= 0.01

        if reward > 0 and abs(obs.head_x - obs.pelvis_x) >= 0.25:
            reward *= 0.25

        return observation, reward, done, info
