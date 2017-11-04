from collections import namedtuple
import numpy as np


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

        self.right_ankle_r = state[6]
        self.right_knee_r = state[7]
        self.right_hip_r = state[8]
        self.left_ankle_r = state[9]
        self.left_knee_r = state[10]
        self.left_hip_r = state[11]

        self.right_ankle_vr = state[12]
        self.right_knee_vr = state[13]
        self.right_hip_vr = state[14]
        self.left_ankle_vr = state[15]
        self.left_knee_vr = state[16]
        self.left_hip_vr = state[17]

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

        self.left_knee_x, self.left_knee_y = self.find_knee(
                self.pelvis_x, self.pelvis_y,
                self.left_talus_x, self.left_talus_y,
                self.left_knee_r,
            )
        self.right_knee_x, self.right_knee_y = self.find_knee(
                self.pelvis_x, self.pelvis_y,
                self.right_talus_x, self.right_talus_y,
                self.right_knee_r,
            )

    def find_knee(self, px, py, tx, ty, kr):
        d = ((py - ty) ** 2 + (px - tx) ** 2) ** 0.5
        d2 = max(0.0, (0.46 ** 2 - (d/2) ** 2)) ** 0.5 * (1.0 if kr <= 0.0 else -1.0)
        x2 = (px + tx) / 2
        y2 = (py + ty) / 2
        x = x2 + (py - ty) / d * d2
        y = y2 + (tx - px) / d * d2
        return x, y


class FeatureGenerator(object):
    def __init__(self):
        self.step = 0
        self.balls = {}
        self.traj = None

    def draw_balls(self, x):
        l = -10
        r = 15
        heights = [0.0 for i in range(l, r)]
        for i in range(l, r):
            for ball in self.balls.values():
                heights[i-l] = max(heights[i-l], ball.height(x + i / 20))
        return heights

    def gen(self, state):
        obs = Observation(state)

        if self.step == 0:
            self.traj = [obs, obs, obs]
        else:
            self.traj = [obs, self.traj[0], self.traj[1]]

        if obs.ball_radius > 0.0:
            ball_abs_x = obs.ball_relative_x + obs.pelvis_x
            self.balls[ball_abs_x] = Ball(ball_abs_x, obs.ball_y, obs.ball_radius)

        central = []
        right = []
        left = []
        extero = []

        central, left, right = [], [], []
        central.append(obs.pelvis_r / 2)
        central.append(obs.pelvis_y - 0.5)
        central.append(obs.pelvis_vr / 2)
        central.append(obs.pelvis_vx / 10)
        central.append(obs.pelvis_vy / 10)

        left.append(obs.left_ankle_r / 4)
        left.append(obs.left_knee_r / 4)
        left.append(obs.left_hip_r / 4)
        right.append(obs.right_ankle_r / 4)
        right.append(obs.right_knee_r / 4)
        right.append(obs.right_hip_r / 4)

        left.append(obs.left_ankle_vr / 4)
        left.append(obs.left_knee_vr / 4)
        left.append(obs.left_hip_vr / 4)
        right.append(obs.right_ankle_vr / 4)
        right.append(obs.right_knee_vr / 4)
        right.append(obs.right_hip_vr / 4)

        central.append(obs.mass_x - obs.pelvis_x)
        central.append(obs.mass_y - obs.pelvis_y)
        central.append((obs.mass_vx - obs.pelvis_vx) / 10)
        central.append((obs.mass_vy - obs.pelvis_vy) / 10)

        central.append(obs.head_x - obs.pelvis_x)
        central.append(obs.head_y - obs.pelvis_y)
        central.append(obs.torso_x - obs.pelvis_x)
        central.append(obs.torso_y - obs.pelvis_y)

        left.append(obs.left_toe_x - obs.pelvis_x)
        left.append(obs.left_toe_y - obs.pelvis_y)
        right.append(obs.right_toe_x - obs.pelvis_x)
        right.append(obs.right_toe_y - obs.pelvis_y)

        left.append(obs.left_talus_x - obs.pelvis_x)
        left.append(obs.left_talus_y - obs.pelvis_y)
        right.append(obs.right_talus_x - obs.pelvis_x)
        right.append(obs.right_talus_y - obs.pelvis_y)

        left.append(obs.left_psoas_str)
        right.append(obs.right_psoas_str)

        central.append(obs.head_y - 0.5)
        central.append(obs.pelvis_y - 0.5)
        central.append(obs.torso_y - 0.5)
        left.append(obs.left_toe_y - 0.5)
        right.append(obs.right_toe_y - 0.5)
        left.append(obs.left_talus_y - 0.5)
        right.append(obs.right_talus_y - 0.5)

        left.append(obs.left_knee_x - obs.pelvis_x)
        left.append(obs.left_knee_y - obs.pelvis_y)
        left.append(obs.left_knee_y - 0.5)
        left.append(obs.left_knee_x - obs.left_talus_x)

        right.append(obs.right_knee_x - obs.pelvis_x)
        right.append(obs.right_knee_y - obs.pelvis_y)
        right.append(obs.right_knee_y - 0.5)
        right.append(obs.right_knee_x - obs.right_talus_x)

        vel = []
        left_vel = []
        right_vel = []

        obs0, obs1 = self.traj[0], self.traj[1]
        pvx = obs0.pelvis_x - obs1.pelvis_x
        pvy = obs0.pelvis_y - obs1.pelvis_y

        vel.append(obs0.head_x - obs1.head_x)
        vel.append(obs0.head_y - obs1.head_y)
        vel.append(obs0.torso_x - obs1.torso_x)
        vel.append(obs0.torso_y - obs1.torso_y)

        left_vel.append(obs0.left_toe_x - obs1.left_toe_x)
        left_vel.append(obs0.left_toe_y - obs1.left_toe_y)
        right_vel.append(obs0.right_toe_x - obs1.right_toe_x)
        right_vel.append(obs0.right_toe_y - obs1.right_toe_y)

        left_vel.append(obs0.left_talus_x - obs1.left_talus_x)
        left_vel.append(obs0.left_talus_y - obs1.left_talus_y)
        right_vel.append(obs0.right_talus_x - obs1.right_talus_x)
        right_vel.append(obs0.right_talus_y - obs1.right_talus_y)

        left_vel.append(obs0.left_knee_x - obs1.left_knee_x)
        left_vel.append(obs0.left_knee_y - obs1.left_knee_y)
        right_vel.append(obs0.right_knee_x - obs1.right_knee_x)
        right_vel.append(obs0.right_knee_y - obs1.right_knee_y)

        def relative_vel(v):
            rvel = []
            for i in range(len(v)):
                if i % 2 == 0:
                    rvel += [v[i] - pvx]
                else:
                    rvel += [v[i] - pvy]
            return rvel

        rvel = relative_vel(vel)
        left_rvel = relative_vel(left_vel)
        right_rvel = relative_vel(right_vel)

        central += [v * 10 for v in rvel]
        left += [v * 10 for v in left_rvel]
        right += [v * 10 for v in right_rvel]

        left += [np.clip(-0.027 - obs.left_toe_y, 0, 0.05) * 20]
        left += [np.clip(0.023 - obs.left_talus_y, 0, 0.05) * 20]

        right += [np.clip(-0.027 - obs.right_toe_y, 0, 0.05) * 20]
        right += [np.clip(0.023 - obs.right_talus_y, 0, 0.05) * 20]

        extero = self.draw_balls(obs.pelvis_x)

        start = [1.0 if self.step < 150 else 0.0, 0.0]
        self.step += 1

        #print(len(central) + len(left) + len(right), len(extero), len(start))
        #print(len(central), len(left), len(right), len(extero), len(start))
        return central + left + right + extero + start


if __name__ == '__main__':
    from osim.env import RunEnv
    env = RunEnv(visualize=False)
    state = env.reset()
    fg = FeatureGenerator()
    fg.gen(state)
