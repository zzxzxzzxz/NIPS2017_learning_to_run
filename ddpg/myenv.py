from osim.env import RunEnv
from feature_generator import Observation

def dist(x0, y0, x1, y1):
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5


class MyRunEnv(RunEnv):

    def step(self, action):
        observation, reward, done, info = super(MyRunEnv, self).step(action)
        curr = Observation(observation)
        prev = Observation(self.last_state)

        pelvis_dx = curr.pelvis_x - prev.pelvis_x
        nz = (curr.head_y - curr.pelvis_y) / dist(curr.head_x, curr.head_y, curr.pelvis_x, curr.pelvis_y)

        if pelvis_dx > 0:
            reward -= (1.0 - nz) * pelvis_dx

        if self.istep >= 70:
            if curr.left_knee_r > 0.015:
                reward -= 0.01
            if curr.right_knee_r > 0.015:
                reward -= 0.01


        return observation, reward, done, info
