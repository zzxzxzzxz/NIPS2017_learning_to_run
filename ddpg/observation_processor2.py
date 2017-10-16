from collections import namedtuple

import numpy as np

Data = namedtuple('Data', ['features', 'observation', 'balls'])


class Observation(object):
    def __init__(self, _state):
        state = list(state)
        self.pelvis_r = state[0]
        self.pelvis_x = state[1]
        self.pelvis_y = state[2]

        self.pelvis_vr = state[3]
        self.pelvis_vx = state[4]
        self.pelvis_vy = state[5]

        self.left_body_r = state[6:9]
        self.right_body_r = state[9:12]

        self.left_body_av = state[12:15]
        self.right_body_av = state[15:18]

        self.mass_x, self.mass_y = state[18], state[19]
        self.mass_vx, self.mass_vy = state[20], state[21]

        self.head_x, self.head_y = state[22], state[23]
        _pelvis_x, _pelvis_y = state[24], state[25]
        assert _pelvis_x == self.pelvis_x
        assert _plevis_y == self.pelvis_y

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


def gen_features(step, state, prev_data=None):
    obs = Observation(state)

    central = []
    right = []
    left = []
    balls = prev_data.balls if prev_data else dict()

    central.append(obs.pelvis_r)
    central.append(obs.pelvis_y - 0.665)
    central.append(obs.head_x - obs.pelvis_x)
    central.append(obs.pelvis_vy / 10)


# 41 dim to 48 dim
def process_observation(observation):
    o = list(observation) # an array

    pr = o[0]

    px = o[1]
    py = o[2]

    pvr = o[3]

    pvx = o[4]
    pvy = o[5]

    for i in range(6,18):
        o[i]/=4

    o = o + [o[22+i*2+1]-0.5 for i in range(7)] # a copy of original y, not relative y.

    # x and y relative to pelvis
    for i in range(7): # head pelvis torso, toes and taluses
        o[22+i*2+0] -= px
        o[22+i*2+1] -= py

    o[18] -= px # mass pos xy made relative
    o[19] -= py
    o[20] -= pvx # mass vel xy made relative
    o[21] -= pvy

    o[38]= min(4,o[38])/3 # ball info are included later in the stage
    # o[39]/=5
    # o[40]/=5

    o[0]/=4 # divide pr by 4
    o[1]=0 # abs value of pel x is not relevant
    o[2]-= 0.5 # minus py by 0.5

    o[3] /=4 # divide pvr by 4
    o[4] /=10 # divide pvx by 10
    o[5] /=10

    o[20]/=10
    o[21]/=10

    return o

_stepsize = 0.01
flatten = lambda l: [item for sublist in l for item in sublist]

# expand observation from 48 to 48*7 dims
processed_dims = 48 + 14*4 + 9*0 + 1*0 + 8
# processed_dims = 41*8
def generate_observation(new, old=None, step=None):

    global _stepsize
    if step is None:
        raise Exception('step should be a valid integer')

    # deal with old
    if old is None:
        if step!=0:
            raise Exception('step nonzero, old == None, how can you do such a thing?')

        old = {'dummy':None,'balls':[],'que':fifo(1200),'last':step-1}
        for i in range(6):
            old['que'].push(new)

    q = old['que']

    if old['last']+1 != step:
        raise Exception('step not monotonically increasing by one')
    else:
        old['last'] += 1

    if step > 1: # bug in osim-rl
        if q.fromtail(0)[36] != new[36]:
            # if last obs and this obs have different psoas value
            print('@step {} Damned'.format(step))
            q.push(['compare(que, new):', q.fromtail(0)[36], new[36]])
            q.dump(reason='obsmixed')
            raise Exception('Observation mixed up, potential bug in parallel code.')

    # q.pop() # remove head
    q.push(new) # add to tail

    # process new
    def lp(n):return list(process_observation(n))
    new_processed = lp(new)

    def bodypart_velocities(at):
        return [(q.fromtail(0+at)[i]-q.fromtail(1+at)[i])/_stepsize for i in range(22,36)]

    def relative_bodypart_velocities(at):
        # velocities, but relative to pelvis.
        bv = bodypart_velocities(at)
        pv1,pv2 = bv[2],bv[3]
        for i in range(len(bv)):
            if i%2==0:
                bv[i] -= pv1
            else:
                bv[i] -= pv2
        return bv

    vels = [bodypart_velocities(k) for k in [0,1]] #[[14][14]]
    relvels = [relative_bodypart_velocities(k) for k in [0,]] #[[14]]
    accs = [
        [
            (vels[t][idx] - vels[t+1][idx])/_stepsize
            for idx in range(len(vels[0]))]
        for t in [0,]]
    # [[14]]

    fv = [v/10 for v in flatten(vels)]
    frv = [rv/10 for rv in flatten(relvels)]
    fa = [a/10 for a in flatten(accs)]
    final_observation = new_processed + fv + frv + fa
    # 48+14*4

    # final_observation += flatten(
    #     [lp(q.fromtail(idx))[38:41] for idx in reversed([4,8,16,32,64])]
    # )
    # # 4 * 5
    # # 48*4

    balls = old['balls']

    def addball_if_new():
        current_pelvis = new[1]
        current_ball_relative = new[38]
        current_ball_height = new[39]
        current_ball_radius = new[40]

        absolute_ball_pos = current_ball_relative + current_pelvis

        if current_ball_radius == 0: # no balls ahead
            return

        compare_result = [abs(b[0] - absolute_ball_pos) < 1e-9 for b in balls]
        # [False, False, False, False] if is different ball

        got_new = sum([(1 if r==True else 0)for r in compare_result]) == 0

        if got_new:
            # for every ball there is
            for b in balls:
                # if this new ball is smaller in x than any ball there is
                if absolute_ball_pos < (b[0] - 1e-9):
                    print(absolute_ball_pos,balls)
                    print('(@ step )'+str(step)+')Damn! new ball closer than existing balls.')
                    q.dump(reason='ballcloser')
                    raise Exception('new ball closer than the old ones.')

            balls.append([
                absolute_ball_pos,
                current_ball_height,
                current_ball_radius,
            ])
            if len(balls)>3:
                print(balls)
                print('(@ step '+str(step)+')What the fuck you just did! Why num of balls became greater than 3!!!')
                q.dump(reason='ballgt3')
                raise Exception('ball number greater than 3.')
        else:
            pass # we already met this ball before.

    if step > 0:
        # initial observation is very wrong, due to implementation bug.
        addball_if_new()

    ball_vectors = []
    current_pelvis = new[1]

    # there should be at most 3 balls
    for i in range(3):
        if i<len(balls):
            rel = balls[i][0] - current_pelvis
            falloff = min(1,max(0,3-abs(rel))) # when ball is closer than 3 falloff become 1
            ball_vectors.append([
                min(4,max(-3, rel))/3, # ball pos relative to current pos
                balls[i][1] * 5 * falloff, # radius
                balls[i][2] * 5 * falloff, # height
            ])
        else:
            ball_vectors.append([
                0,
                0,
                0,
            ])

    # 9-d
    # final_observation += flatten(reversed(ball_vectors))

    # episode_end_indicator = max(0, (step/1000-0.6))/10 # lights up when near end-of-episode
    # final_observation[1] = episode_end_indicator
    #
    # final_observation += [episode_end_indicator]

    # flat_ahead_indicator = np.clip((current_pelvis - 5.0)/2, 0.0, 1.0)
    # # 0 at 5m, 1 at 7m
    #
    # final_observation += [flat_ahead_indicator]

    foot_touch_indicators = []
    for i in [29,31,33,35]: # y of toes and taluses
        # touch_ind = 1 if new[i] < 0.05 else 0
        touch_ind = np.clip(0.05 - new[i] * 10 + 0.5, 0., 1.)
        touch_ind2 = np.clip(0.1 - new[i] * 10 + 0.5, 0., 1.)
        # touch_ind2 = 1 if new[i] < 0.1 else 0
        foot_touch_indicators.append(touch_ind)
        foot_touch_indicators.append(touch_ind2)
    final_observation+=foot_touch_indicators # 8dim

    # for i,n in enumerate(new_processed):
    #     print(i,n)

    def final_processing(l):
        # normalize to prevent excessively large input
        for idx in range(len(l)):
            if l[idx] > 1: l[idx] = np.sqrt(l[idx])
            if l[idx] < -1: l[idx] = - np.sqrt(-l[idx])
    final_processing(final_observation)

    return final_observation, old

if __name__=='__main__':
    ff = fifo(4)
    ff.push(1)
    ff.push(3)
    ff.push(5)
    ff.pop()
    ff.pop()
    ff.push(6)
    ff.push(7)

    print(ff.fromhead(0))
    print(ff.fromhead(1))
    print(ff.fromtail(0))
    print(ff.fromtail(1))
