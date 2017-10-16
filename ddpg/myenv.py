from osim.env import RunEnv


class MyRunEnv(RunEnv):

    def step(self, action):
        observation, reward, done, info = super(MyRunEnv, self).step(action)

        pelvis_y = observation[2]

        if pelvis_y < 0.665:
            done = True

        return observation, reward, done, info
