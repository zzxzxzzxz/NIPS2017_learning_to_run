import sys
import os
import time
import pickle
import argparse
import random
from threading import Lock, Thread
from copy import deepcopy

import gym
from osim.env import RunEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from osim.http.client import Client

from noise import one_fsq_noise
#from multi import fastenv
#from observation_processor import generate_observation as go
from feature_generator2 import FeatureGenerator
from wrap_env import fastenv
from plotter import interprocess_plotter as Plotter

np.random.seed(314)
torch.manual_seed(314)

#####################  hyper parameters  ####################

USE_CUDA = True
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
MAX_EP_STEPS = 1000
ENV_SKIP = 2
LR_ACTOR = 1e-4     # learning rate for actor
LR_CRITIC = 3e-4    # learning rate for critic
GAMMA = 0.99        # reward discount
TAU = 1e-3
MEMORY_CAPACITY = 500000
BATCH_SIZE = 64
TOKEN = '0f3e16541bd585c72ccb1ad840807d7f'

DIM_CENT = 20
DIM_LEFT = 25
DIM_RIGHT = 25
DIM_EX = 25
DIM_START = 1
DIM_ZERO = 1
DIM_ACTION = 9
OFFSET_CENT = DIM_CENT
OFFSET_LEFT = OFFSET_CENT + DIM_LEFT
OFFSET_RIGHT = OFFSET_LEFT + DIM_RIGHT
OFFSET_EX = OFFSET_RIGHT + DIM_EX
OFFSET_START = OFFSET_EX + DIM_START
DIM_ALL = OFFSET_START + DIM_ZERO


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.extero1 = nn.Linear(DIM_EX, 64)
        self.extero2 = nn.Linear(64, 64)
        self.hidden1 = nn.Linear(DIM_CENT + DIM_LEFT + DIM_RIGHT, 256)
        self.hidden2 = nn.Linear(256, 128)

        #self.start = nn.Linear(1, 64)
        self.hidden3 = nn.Linear(128 + 64, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.hidden5 = nn.Linear(128, DIM_ACTION)

        self.ln_e1 = LayerNorm(64)
        self.ln_e2 = LayerNorm(64)

        self.ln_h1 = LayerNorm(256)
        self.ln_h2 = LayerNorm(128)
        self.ln_h3 = LayerNorm(128)
        self.ln_h4 = LayerNorm(128)

    def forward(self, x):
        cent = x[:, :OFFSET_CENT]
        left = x[:, OFFSET_CENT:OFFSET_LEFT]
        right = x[:, OFFSET_LEFT:OFFSET_RIGHT]
        extero = x[:, OFFSET_RIGHT:OFFSET_EX]
        #start = x[:, OFFSET_EX:OFFSET_START]
        #zero = x[:, OFFSET_START:]

        #start = self.start(start)
        #zero = self.start(zero)

        x1 = torch.cat([cent, left, right], 1)
        x2 = torch.cat([cent, right, left], 1)
        x3 = extero

        x1 = F.leaky_relu(self.hidden1(x1), negative_slope=0.2)
        x1 = self.ln_h1(x1)
        x1 = F.leaky_relu(self.hidden2(x1), negative_slope=0.2)
        x1 = self.ln_h2(x1)

        x2 = F.leaky_relu(self.hidden1(x2), negative_slope=0.2)
        x2 = self.ln_h1(x2)
        x2 = F.leaky_relu(self.hidden2(x2), negative_slope=0.2)
        x2 = self.ln_h2(x2)

        x3 = F.leaky_relu(self.extero1(x3), negative_slope=0.2)
        x3 = self.ln_e1(x3)
        x3 = F.leaky_relu(self.extero2(x3), negative_slope=0.2)
        x3 = self.ln_e2(x3)

        x4 = torch.cat([x1, x3], 1)
        x5 = torch.cat([x2, x3], 1)

        x4 = F.leaky_relu(self.hidden3(x4), negative_slope=0.2)
        x4 = self.ln_h3(x4)
        x4 = F.leaky_relu(self.hidden4(x4), negative_slope=0.2)
        x4 = self.ln_h4(x4)
        x4 = F.tanh(self.hidden5(x4)) * 0.5 + 0.5

        x5 = F.leaky_relu(self.hidden3(x5), negative_slope=0.2)
        x5 = self.ln_h3(x5)
        x5 = F.leaky_relu(self.hidden4(x5), negative_slope=0.2)
        x5 = self.ln_h4(x5)
        x5 = F.tanh(self.hidden5(x5)) * 0.5 + 0.5

        x = torch.cat([x4, x5], 1)
        return x

    def init_weights(self):
        super(Actor, self).init_weights()
        #self.start.weight.data.normal_(0.0, 1.0)
        self.extero1.weight.data.normal_(0.0, 1.66667)
        self.extero2.weight.data.normal_(0.0, 1.66667)
        self.hidden1.weight.data.normal_(0.0, 1.66667)
        self.hidden2.weight.data.normal_(0.0, 1.66667)
        self.hidden3.weight.data.normal_(0.0, 1.66667)
        self.hidden4.weight.data.normal_(0.0, 1.66667)
        self.hidden5.weight.data.normal_(0.0, 1.0)


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(DIM_ALL, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128 + DIM_ACTION * 2, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.hidden5 = nn.Linear(128, 1)
        self.ln1 = LayerNorm(256)
        self.ln2 = LayerNorm(128)
        self.ln3 = LayerNorm(128)
        self.ln4 = LayerNorm(128)

    def forward(self, x):
        obs, act = x
        x = F.leaky_relu(self.hidden1(obs), negative_slope=0.2)
        x = self.ln1(x)
        x = F.leaky_relu(self.hidden2(x), negative_slope=0.2)
        x = self.ln2(x)
        x = torch.cat([x, act], 1)
        x = F.leaky_relu(self.hidden3(x), negative_slope=0.2)
        x = self.ln3(x)
        x = F.leaky_relu(self.hidden4(x), negative_slope=0.2)
        x = self.ln4(x)
        x = self.hidden5(x)
        return x

    def init_weights(self):
        super(Critic, self).init_weights()
        self.hidden1.weight.data.normal_(0.0, 1.66667)
        self.hidden2.weight.data.normal_(0.0, 1.66667)
        self.hidden3.weight.data.normal_(0.0, 1.66667)
        self.hidden4.weight.data.normal_(0.0, 1.66667)
        self.hidden5.weight.data.normal_(0.0, 1.0)


def sync_target(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(FLOAT)


def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()


def deepcopy_all(*args):
    return list(map(deepcopy, args))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.lock = Lock()

    def push(self, data):
        with self.lock:
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = data
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def save(self, path):
        with self.lock:
            with open(path, 'wb') as f:
                pickle.dump([self.memory ,self.position], f)
            print('memory dumped into', path)

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory, self.position = pickle.load(f)
        print('memory loaded from', path)


class DDPG(object):
    def __init__(self):
        self.tau = TAU
        self.discount = GAMMA
        self.batch_size = BATCH_SIZE

        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_optim = Adam(self.actor.parameters(), lr=LR_ACTOR, weight_decay=1e-7)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optim = Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=1e-7)

        sync_target(self.actor_target, self.actor, 1.0)
        sync_target(self.critic_target, self.critic, 1.0)

        if USE_CUDA:
            self.cuda()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def select_action(self, state):
        action = self.actor(to_tensor(np.array([state])))
        action = to_numpy(action).squeeze(0)
        return action

    def learn(self):
        batch = self.memory.sample(self.batch_size)
        state, action, reward, next_state, done = list(map(np.array, zip(*batch)))

        # Prepare for the target q
        next_q = self.critic_target([
            to_tensor(next_state, volatile=True),
            self.actor_target(to_tensor(next_state, volatile=True)),
        ])
        next_q.volatile = False
        target_q = to_tensor(reward) + self.discount * to_tensor(1.0 - done) * next_q

        # Critic update
        self.critic.zero_grad()
        q = self.critic([to_tensor(state), to_tensor(action)])
        critic_loss = self.criterion(q, target_q)
        critic_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        actor_loss = -self.critic([to_tensor(state), self.actor(to_tensor(state))]).mean()
        actor_loss.backward()
        self.actor_optim.step()

        # Target update
        sync_target(self.actor_target, self.actor, self.tau)
        sync_target(self.critic_target, self.critic, self.tau)

    def load_model(self, path, load_memory=True):
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(path))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(path))
        )
        self.actor_target.load_state_dict(
            torch.load('{}/actor_target.pkl'.format(path))
        )
        self.critic_target.load_state_dict(
            torch.load('{}/critic_target.pkl'.format(path))
        )
        if load_memory:
            self.memory.load('{}/rpm.pkl'.format(path))

    def save_model(self, path, save_memory=True):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(path)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(path)
        )
        torch.save(
            self.actor_target.state_dict(),
            '{}/actor_target.pkl'.format(path)
        )
        torch.save(
            self.critic_target.state_dict(),
            '{}/critic_target.pkl'.format(path)
        )
        if save_memory:
            self.memory.save('{}/rpm.pkl'.format(path))


class DistributedTrain(object):

    def __init__(self, agent):
        self.agent = agent
        self.lock = Lock()
        self.plotter = Plotter(num_lines=3)

        from farmer import farmer as farmer_class
        self.farmer = farmer_class()


    def playonce(self, noise_level, _env):
        t = time.time()

        skip = ENV_SKIP
        env = fastenv(_env, skip)

        noise_source1 = one_fsq_noise()
        noise_source2 = one_fsq_noise()
        for j in range(200):
            noise_source1.one((DIM_ACTION,), noise_level)
            noise_source2.one((DIM_ACTION,), noise_level)

        r1 = min(1.0, noise_level - 0.1)
        r2 = 1.0 - r1

        state = env.reset()

        n_steps = 0
        ep_reward = 0
        warmup = BATCH_SIZE * 32

        noise_phase = int(np.random.uniform() * 999999)

        while True:
            action = self.agent.select_action(state)

            phased_noise_anneal_duration = 100
            phased_noise_amplitude = ((-noise_phase-n_steps) % phased_noise_anneal_duration) / phased_noise_anneal_duration
            phased_noise_amplitude = max(0, phased_noise_amplitude * 2 - 1)
            phased_noise_amplitude = max(0.01, phased_noise_amplitude ** 2)

            exploration_noise1 = noise_source1.one((DIM_ACTION,), noise_level) * phased_noise_amplitude
            exploration_noise2 = noise_source2.one((DIM_ACTION,), noise_level) * phased_noise_amplitude
            exploration_noise = np.concatenate([exploration_noise1, exploration_noise1 * r1 + exploration_noise2 * r2])
            action += exploration_noise * 0.5 * phased_noise_amplitude
            action = np.clip(action, 0, 1)

            next_state, reward, done, info = env.step(action.tolist())
            self.agent.memory.push(deepcopy_all(state, action, [reward], next_state, [float(done)]))

            if len(self.agent.memory) >= warmup:
                with self.lock:
                    self.agent.learn()

            state = next_state
            ep_reward += reward
            n_steps += 1

            if done:
                break

        with self.lock:
            t = time.time() - t
            print('reward: {}, n_steps: {}, explore: {:.5f}, n_mem: {}, time: {:.2f}' \
                  .format(ep_reward, n_steps, noise_level, len(self.agent.memory), t))

            global t0
            self.plotter.pushys([ep_reward, noise_level, (time.time() - t0) % 3600 / 3600 - 2])

        _env.rel()
        del env

    def play_if_available(self, noise_level):
        while True:
            remote_env = self.farmer.acq_env()
            if remote_env:
                t = Thread(target=self.playonce, args=(noise_level, remote_env), daemon=True)
                t.start()
                break
            else:
                time.sleep(0.005)


def train(args):
    print('start training')
    global t0
    t0 = time.time()

    ddpg = DDPG()

    if args.resume > 0:
        print('load model {}/{}'.format(args.model, args.resume))
        ddpg.load_model('{}/{}'.format(args.model, args.resume))

    dist_train = DistributedTrain(ddpg)

    noise_decay_rate = 0.0005
    noise_floor = 0.1
    noiseless = 0.01
    noise_level = 1.2 * ((1.0 - noise_decay_rate) ** args.resume)

    for i in range(args.resume, args.max_ep):
        print('Episode {} / {}'.format(i + 1, args.max_ep))

        noise_level *= (1.0 - noise_decay_rate)
        noise_level = max(noise_floor, noise_level)

        nl = noise_level if (i + 1) % 20 else noiseless
        dist_train.play_if_available(nl)

        print('elapsed time: {0:.2f} secs'.format(time.time() - t0))
        sys.stdout.flush()
        time.sleep(0.005)

        if args.model and (i + 1) % 100 == 0:
            ddpg.save_model('{}/{}'.format(args.model, i + 1))


def test(args):
    print('start testing')

    ddpg = DDPG()
    ddpg.load_model(args.model, load_memory=False)
    env = RunEnv(visualize=args.visualize, max_obstacles=3)

    np.random.seed(1131231)
    for i in range(1):
        step = 0
        state = env.reset(difficulty=2)
        fg = FeatureGenerator()

        state = fg.gen(state)
        obs = fg.traj[0]
        #print(obs.left_knee_x, obs.left_talus_x)

        ep_reward = 0
        ep_memories = []
        while True:
            action = ddpg.select_action(list(state))
            next_state, reward, done, info = env.step(action.tolist())
            next_state = fg.gen(next_state)

#            obs = fg.traj[0]
#            print(obs.left_knee_r, obs.left_ankle_r, obs.left_hip_r, obs.left_toe_y)
#            print(obs.right_knee_r, obs.right_ankle_r, obs.right_hip_r, obs.right_toe_y)

            print('step: {0:03d}'.format(step), end=', action: ')
            for act in action:
                print('{0:.3f}'.format(act), end=', ')
            print()

            state = next_state
            ep_reward += reward
            step += 1

            if done:
                break

        print('\nEpisode: {} Reward: {}, n_steps: {}'.format(i, ep_reward, step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--max_ep', default=10000, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--resume', default=0, type=int)

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true')
    action.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)
