from collections import namedtuple
import numpy as np


class FeatureGenerator(object):
    dim_feature = 112

    def __init__(self):
        self.step = 0
        self.balls = {}
        self.vision = 0
        self.traj = None

    def draw_balls(self, x):
        l = 0
        r = 15
        heights = [0.0 for i in range(l, r)]
        invisible = [0.0 for i in range(l, r)]
        for i in range(l, r):
            invisible[i-l] = 1.0 if x + i / 10 >= self.vision else 0.0
            for ball in self.balls.values():
                heights[i-l] = max(heights[i], ball.height(x + i / 10))
        return list(zip(heights, invisible))

    def gen(self, state):
        obs = Observation(state)

        if self.step == 0:
            self.traj = [obs, obs, obs]
        else:
            self.traj = [obs, self.traj[0], self.traj[1]]

        central = []
        right = []
        left = []
        extero = []

        f = []
        f.append(obs.pelvis_r / 4)
        f.append(0)
        f.append(obs.pelvis_y - 0.5)
        f.append(obs.pelvis_vr / 4)
        f.append(obs.pelvis_vx / 10)
        f.append(obs.pelvis_vy / 10)

        f += [r / 4 for r in obs.left_body_rs]
        f += [r / 4 for r in obs.right_body_rs]
        f += [av / 4 for av in obs.left_body_avs]
        f += [av / 4 for av in obs.right_body_avs]

        f.append(obs.mass_x - obs.pelvis_x)
        f.append(obs.mass_y - obs.pelvis_y)
        f.append((obs.mass_vx - obs.pelvis_vx) / 10)
        f.append((obs.mass_vy - obs.pelvis_vy) / 10)

        f.append(obs.head_x - obs.pelvis_x)
        f.append(obs.head_y - obs.pelvis_y)
        f.append(0)
        f.append(0)
        f.append(obs.torso_x - obs.pelvis_x)
        f.append(obs.torso_y - obs.pelvis_y)

        f.append(obs.left_toe_x - obs.pelvis_x)
        f.append(obs.left_toe_y - obs.pelvis_y)
        f.append(obs.right_toe_x - obs.pelvis_x)
        f.append(obs.right_toe_y - obs.pelvis_y)

        f.append(obs.left_talus_x - obs.pelvis_x)
        f.append(obs.left_talus_y - obs.pelvis_y)
        f.append(obs.right_talus_x - obs.pelvis_x)
        f.append(obs.right_talus_y - obs.pelvis_y)

        f.append(obs.left_psoas_str)
        f.append(obs.right_psoas_str)

        f.append(min(4, obs.ball_relative_x) / 3)
        f.append(obs.ball_y)
        f.append(obs.ball_radius)

        f.append(obs.head_y - 0.5)
        f.append(obs.pelvis_y - 0.5)
        f.append(obs.torso_y - 0.5)
        f.append(obs.left_toe_y - 0.5)
        f.append(obs.right_toe_y - 0.5)
        f.append(obs.left_talus_y - 0.5)
        f.append(obs.right_talus_y - 0.5)

        vel, rvel = [[], []], []
        for t in range(2):
            obs0, obs1 = self.traj[t], self.traj[t+1]
            vel[t].append(obs0.head_x - obs1.head_x)
            vel[t].append(obs0.head_y - obs1.head_y)
            vel[t].append(obs0.pelvis_x - obs1.pelvis_x)
            vel[t].append(obs0.pelvis_y - obs1.pelvis_y)
            vel[t].append(obs0.torso_x - obs1.torso_x)
            vel[t].append(obs0.torso_y - obs1.torso_y)

            vel[t].append(obs0.left_toe_x - obs1.left_toe_x)
            vel[t].append(obs0.left_toe_y - obs1.left_toe_y)
            vel[t].append(obs0.right_toe_x - obs1.right_toe_x)
            vel[t].append(obs0.right_toe_y - obs1.right_toe_y)

            vel[t].append(obs0.left_talus_x - obs1.left_talus_x)
            vel[t].append(obs0.left_talus_y - obs1.left_talus_y)
            vel[t].append(obs0.right_talus_x - obs1.right_talus_x)
            vel[t].append(obs0.right_talus_y - obs1.right_talus_y)

        vel[0] = [v / 0.01 for v in vel[0]]
        vel[1] = [v / 0.01 for v in vel[1]]

        pvx, pvy = vel[0][2], vel[0][3]
        def relative_vel(v):
            rvel = []
            for i in range(len(v)):
                if i % 2 == 0:
                    rvel += [v[i] - pvx]
                else:
                    rvel += [v[i] - pvy]
            return rvel
        rvel = relative_vel(vel[0])
        acc = [(v0 - v1) / 0.01 for v0, v1 in zip(vel[0], vel[1])]

        vel[0] = [v / 10 for v in vel[0]]
        vel[1] = [v / 10 for v in vel[1]]
        rvel = [v / 10 for v in rvel]
        acc = [v / 10 for v in acc]

        f += vel[0] + vel[1] + rvel + acc

        f += [np.clip(0.05 - obs.left_toe_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.1 - obs.left_toe_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.05 - obs.right_toe_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.1 - obs.right_toe_y * 10 + 0.5, 0.0, 1.0)]

        f += [np.clip(0.05 - obs.left_talus_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.1 - obs.left_talus_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.05 - obs.right_talus_y * 10 + 0.5, 0.0, 1.0)]
        f += [np.clip(0.1 - obs.right_talus_y * 10 + 0.5, 0.0, 1.0)]

        for i in range(len(f)):
            if f[i] > 1:
                f[i] = np.sqrt(f[i])
            if f[i] < -1:
                f[i] = -np.sqrt(-f[i])

        self.step += 1

        return f


class Ball(object):
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def height(self, sample_x):
        eps = 0.001
        if sample_x >= self.x + self.radius - eps or sample_x <= self.x - self.radius + eps:
            return 0.0

        h = (self.radius ** 2 - (sample_x - self.x) ** 2) ** 0.5
        return max(0.0, h + self.y)


class Observation(object):
    def __init__(self, _state):
        state = list(_state)
        self.pelvis_r = state[0]
        self.pelvis_x = state[1]
        self.pelvis_y = state[2]

        self.pelvis_vr = state[3]
        self.pelvis_vx = state[4]
        self.pelvis_vy = state[5]

        self.left_body_rs = state[6:9]
        self.right_body_rs = state[9:12]

        self.left_body_avs = state[12:15]
        self.right_body_avs = state[15:18]

        self.mass_x, self.mass_y = state[18], state[19]
        self.mass_vx, self.mass_vy = state[20], state[21]

        self.head_x, self.head_y = state[22], state[23]
        _pelvis_x, _pelvis_y = state[24], state[25]
        assert _pelvis_x == self.pelvis_x
        assert _pelvis_y == self.pelvis_y

        self.torso_x, self.torso_y = state[26], state[27]

        self.left_toe_x, self.left_toe_y = state[28], state[29]
        self.right_toe_x, self.right_toe_y = state[30], state[31]

        self.left_talus_x, self.left_talus_y = state[32], state[33]
        self.right_talus_x, self.right_talus_y = state[34], state[35]

        self.left_psoas_str = state[36]
        self.right_psoas_str = state[37]

        self.ball_relative_x = state[38]
        self.ball_y = state[39]
        self.ball_radius = state[40]
