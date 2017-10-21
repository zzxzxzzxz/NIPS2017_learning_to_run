from osim.env import RunEnv


class MyRunEnv(RunEnv):

    def step(self, action):
        observation, reward, done, info = super(MyRunEnv, self).step(action)

        reward -= observation[1] - self.last_state[1]
        reward += min(observation[1], observation[22] + 0.1) \
                  -  min(self.last_state[1], self.last_state[22] + 0.1)

        return observation, reward, done, info
