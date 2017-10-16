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
from multi import fastenv
from observation_processor import generate_observation as go

np.random.seed(312)
torch.manual_seed(312)

#####################  hyper parameters  ####################

MAX_EP_STEPS = 1000
ENV_SKIP = 2
LR_ACTOR = 1e-4     # learning rate for actor
LR_CRITIC = 3e-4    # learning rate for critic
GAMMA = 0.99        # reward discount
TAU = 5e-4
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 64
TOKEN = '0f3e16541bd585c72ccb1ad840807d7f'


class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        self.hidden1 = nn.Linear(dim_state, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.hidden5 = nn.Linear(128, dim_action)

    def forward(self, x):
        x = F.leaky_relu(self.hidden1(x), negative_slope=0.2)
        x = F.leaky_relu(self.hidden2(x), negative_slope=0.2)
        x = F.leaky_relu(self.hidden3(x), negative_slope=0.2)
        x = F.leaky_relu(self.hidden4(x), negative_slope=0.2)
        x = F.tanh(self.hidden5(x)) * 0.5 + 0.5
        return x

    def init_weights(self):
        self.hidden1.weight.data.normal_(0.0, 1.66667)
        self.hidden2.weight.data.normal_(0.0, 1.66667)
        self.hidden3.weight.data.normal_(0.0, 1.66667)
        self.hidden4.weight.data.normal_(0.0, 0.125)
        self.hidden5.weight.data.normal_(0.0, 1.0)


class Critic(nn.Module):

    def __init__(self, dim_state, dim_action):
        super(Critic, self).__init__()
        self.hidden1 = nn.Linear(dim_state, 256)
        self.hidden2 = nn.Linear(256, 128)
        self.hidden3 = nn.Linear(128 + dim_action, 128)
        self.hidden4 = nn.Linear(128, 128)
        self.hidden5 = nn.Linear(128, 48)
        self.hidden6 = nn.Linear(48, 1)

    def forward(self, x):
        obs, act = x
        x = F.leaky_relu(self.hidden1(obs), negative_slope=0.2)
        x = F.leaky_relu(self.hidden2(x), negative_slope=0.2)
        x = torch.cat([x, act], 1)
        x = F.leaky_relu(self.hidden3(x), negative_slope=0.2)
        x = F.leaky_relu(self.hidden4(x), negative_slope=0.2)
        x = F.leaky_relu(self.hidden5(x), negative_slope=0.2)
        x = self.hidden6(x)
        return x

    def init_weights(self):
        self.hidden1.weight.data.normal_(0.0, 1.66667)
        self.hidden2.weight.data.normal_(0.0, 1.66667)
        self.hidden3.weight.data.normal_(0.0, 1.66667)
        self.hidden4.weight.data.normal_(0.0, 1.66667)
        self.hidden5.weight.data.normal_(0.0, 1.66667)
        self.hidden6.weight.data.normal_(0.0, 1.0)


def sync_target(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(torch.FloatTensor)


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
            self.memory[self.position] = list(map(deepcopy, data))
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
    def __init__(self, dim_state, dim_action):
        self.tau = TAU
        self.discount = GAMMA
        self.batch_size = BATCH_SIZE

        self.dim_state = dim_state
        self.dim_action = dim_action

        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(MEMORY_CAPACITY)

        self.actor = Actor(dim_state, dim_action)
        self.actor_target = Actor(dim_state, dim_action)
        self.actor_optim = Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(dim_state, dim_action)
        self.critic_target = Critic(dim_state, dim_action)
        self.critic_optim = Adam(self.critic.parameters(), lr=LR_CRITIC)

        sync_target(self.actor_target, self.actor, 1.0)
        sync_target(self.critic_target, self.critic, 1.0)

    def select_action(self, state):
        action = self.actor(to_tensor(np.array([state])))
        action = action.data.numpy().squeeze(0)
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

    def load_model(self, path):
        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(path))
        )
        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(path))
        )
        self.memory.load('{}/rpm.pkl'.format(path))

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(path)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(path)
        )
        self.memory.save('{}/rpm.pkl'.format(path))


class DistributedTrain(object):

    def __init__(self, agent):
        self.agent = agent
        self.lock = Lock()

        from farmer import farmer as farmer_class
        self.farmer = farmer_class()

    def playonce(self, noise_level, _env):
        skip = ENV_SKIP
        env = fastenv(_env, skip)
        #max_steps = int(MAX_EP_STEPS / skip)
        max_steps = 50000

        noise_source = one_fsq_noise()
        for j in range(200):
            noise_source.one((self.agent.dim_action,), noise_level)

        state = env.reset()

        ep_memories = []
        n_steps = 0
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            n_steps += 1
            action = self.agent.select_action(state)

            # Add exploration noise
            exploration_noise = noise_source.one((self.agent.dim_action,), noise_level)
            action += exploration_noise * 0.5
            action = np.clip(action, 0, 1)

            next_state, reward, done, info = env.step(action.tolist() + action.tolist())

            ep_memories.append(list(map(deepcopy, [state, action, [reward], next_state, [done]])))
            #self.agent.memory.push([state, action, [reward], next_state, [done]])

            if len(self.agent.memory) > self.agent.batch_size * 128:
                with self.lock:
                    self.agent.learn()

            state = next_state
            ep_reward += reward

            if done:
                break

        for ep_m in ep_memories:
            if np.random.uniform() > 0.5:
                self.agent.memory.push(ep_m)

        print('reward: {}, n_steps: {}, explore: {}, n_mem: {}'.format(ep_reward, n_steps, noise_level, len(self.agent.memory)))

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
                time.sleep(0.01)


def train(args):
    print('start training')
    #env = gym.make('Pendulum-v0')
    #env = RunEnv(visualize=False)
    from observation_processor import processed_dims

    dim_state = processed_dims
    dim_action = 9
    print(dim_state, dim_action)

    ddpg = DDPG(dim_state, dim_action)
#    ddpg.memory.load('try_reproduce/rpm.pkl')
#    for i in range(5000):
#        if (i + 1) % 100 == 0:
#            print(i)
#        ddpg.learn()
    dist_train = DistributedTrain(ddpg)

    noise_level = 2.
    #noise_decay_rate = 0.001
    noise_decay_rate = 0.0005
    noise_floor = 0.05
    noiseless = 0.01

    t = time.time()
    for i in range(args.max_ep):
        print('Episode {} / {}'.format(i + 1, args.max_ep))

        noise_level *= (1.0 - noise_decay_rate)
        noise_level = max(noise_floor, noise_level)

        nl = noise_level if np.random.uniform() > 0.05 else noiseless
        dist_train.play_if_available(nl)

        print('elapsed time: {0:.02f} secs'.format(time.time() - t))
        sys.stdout.flush()
        time.sleep(0.5)

        if args.model and (i + 1) % 200 == 0:
            ddpg.save_model(args.model)


def test(args):
    print('start testing')
    env = RunEnv(visualize=True)
    from observation_processor import processed_dims

    dim_state = processed_dims
    dim_action = 9
    print(dim_state, dim_action)

    ddpg = DDPG(dim_state, dim_action)
    ddpg.load_model(args.model)

    np.random.seed(543)
    for i in range(1):
        step = 0
        state = env.reset(difficulty=2)
        old_state = None
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            state, old_state = go(state, old_state, step=step)

            action = ddpg.select_action(list(state))
            action = [act if act >= 0.5 else 0.0 for act in action.tolist() + action.tolist()]
            #action = [act for act in action.tolist() + action.tolist()]
            next_state, reward, done, info = env.step(action)

            if j == 800:
                for k, obs in enumerate(next_state):
                    print(k, obs)
                print('reward:', ep_reward + reward)
                break

            print('step: {0:03d}'.format(step), end=', action: ')
            for act in action:
                print('{0:.03f}'.format(act), end=', ')
            print()

            state = next_state
            ep_reward += reward
            step += 1

            if done:
                break

        print('Episode: {} Reward: {}, n_steps: {}'.format(i, ep_reward, step))


def submit(args):
    print('start submitting')

    remote_base = 'http://grader.crowdai.org:1729'
    client = Client(remote_base)

    from observation_processor import processed_dims
    dim_state = processed_dims
    dim_action = 9
    print(dim_state, dim_action)

    ddpg = DDPG(dim_state, dim_action)
    ddpg.load_model(args.model)

    state = client.env_create(TOKEN)
    step = 0
    old_state = None
    ep_reward = 0

    while True:
        state, old_state = go(state, old_state, step=step)

        print('selecting action ...', end=' ')
        action = ddpg.select_action(list(state))
        action = [act for act in action.tolist() + action.tolist()]
        print('client.env_step ...')
        next_state, reward, done, info = client.env_step(action)

        print('step: {0:03d}, ep_reward: {1:02.08f}'.format(step, ep_reward))

        state = next_state
        ep_reward += reward
        step += 1

        if done:
            print('done')
            state = client.env_reset()
            if not state:
                break
            step = 0
            old_state = None
            ep_reward = 0

    client.submit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model', type=str)
    parser.add_argument('--max_ep', default=5000, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()

    if args.submit and args.test:
        submit(args)
    elif args.test:
        test(args)
    else:
        train(args)
