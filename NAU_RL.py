"""
Helper functions for RL cartpole simulation
"""

import math
import base64
import argparse
import random
from pathlib import Path
from collections import namedtuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

import stable_nalu

import gym
from gym.wrappers import Monitor
from pprint import pprint
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output

from RLfunctions import *


class NetworkNAURNN(stable_nalu.abstract.ExtendedTorchModule):
    UNIT_NAMES = stable_nalu.layer.GeneralizedCell.UNIT_NAMES

    def __init__(self, unit_name, input_size=5, hidden_size=16, output_size=1, **kwargs):
        super().__init__('network', writer=None, **kwargs)

        self.unit_name = unit_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Since for the 'mul' problem, the zero_state should be 1, and for the
        # 'add' problem it should be 0. The zero_states are allowed to be
        # # optimized.
        if unit_name == 'LSTM':
            self.zero_state = torch.nn.ParameterDict({
                'h_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size)),
                'c_t0': torch.nn.Parameter(torch.Tensor(self.hidden_size))
            })
        else:
            self.zero_state = torch.nn.Parameter(torch.Tensor(self.hidden_size))

        self.recurent_cell = stable_nalu.layer.GeneralizedCell(input_size, self.hidden_size,
                                                               unit_name,
                                                               writer=self.writer,
                                                               name='recurrent_layer',
                                                               **kwargs)
        self.output_layer = stable_nalu.layer.GeneralizedLayer(self.hidden_size, output_size,
                                                               'linear'
                                                               if unit_name in {'GRU', 'LSTM', 'RNN-tanh', 'RNN-ReLU'}
                                                               else unit_name,
                                                               writer=self.writer,
                                                               name='output_layer',
                                                               **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        if self.unit_name == 'LSTM':
            for zero_state in self.zero_state.values():
                torch.nn.init.zeros_(zero_state)
        else:
            torch.nn.init.zeros_(self.zero_state)

        self.recurent_cell.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, x):
        """Performs recurrent iterations over the input.

        Arguments:
            input: Expected to have the shape [obs, time, dims]
        """
        if len(x.size()) == 2:
            x = x[None]
        # Perform recurrent iterations over the input
        if self.unit_name == 'LSTM':
            h_tm1 = tuple(zero_state.repeat(x.size(0), 1) for zero_state in self.zero_state.values())
        else:
            h_tm1 = self.zero_state.repeat(x.size(0), 1)

        hs = []
        for t in range(x.size(1)):
            x_t = x[:, t]
            h_t = self.recurent_cell(x_t, h_tm1)
            h_tm1 = h_t
            hs.append(h_t)

        # Grap the final hidden output and use as the output from the recurrent layer
        #         z_1 = h_t[0] if self.unit_name == 'LSTM' else h_t
        z = [x[0] if self.unit_name == 'LSTM' else x for x in hs]
        #         z_2 = self.output_layer(z_1)
        z_2 = self.output_layer(torch.stack(z))
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )

    def predict(self, x):
        return self(x).detach().numpy()


class NetworkNAU(stable_nalu.abstract.ExtendedTorchModule):
    UNIT_NAMES = stable_nalu.layer.GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_name, input_size=100, hidden_size=2, first_layer=None, nac_mul='none',
                 output_size=1, eps=1e-7, writer=None, actor=False, **kwargs):
        super().__init__('network', writer=writer, **kwargs)
        self.unit_name = unit_name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.nac_mul = nac_mul
        self.eps = eps
        self.actor = actor

        if hasattr(self.hidden_size, '__iter__'):
            self.layers = [stable_nalu.layer.GeneralizedLayer(input_size, hidden_size[0],
                                                              unit_name, writer=self.writer, name='layer_1',
                                                              eps=eps, **kwargs)]
            for i in range(len(hidden_size) - 1):
                self.layers.append(
                    stable_nalu.layer.GeneralizedLayer(hidden_size[i], hidden_size[i + 1],
                                                       unit_name, writer=self.writer, name=f'layer_{i + 2}',
                                                       eps=eps, **kwargs)
                )
            self.layers.append(
                stable_nalu.layer.GeneralizedLayer(hidden_size[-1], output_size,
                                                   'linear', writer=self.writer, name=f'layer_{i + 2}',
                                                   eps=eps, **kwargs)
            )

        else:
            if first_layer is not None:
                unit_name_1 = first_layer
            else:
                unit_name_1 = unit_name
            if nac_mul == 'mnac':
                unit_name_2 = unit_name[0:-3] + 'MNAC'
            else:
                unit_name_2 = unit_name

            self.layer_1 = stable_nalu.layer.GeneralizedLayer(input_size, hidden_size,
                                                              unit_name_1,
                                                              writer=self.writer,
                                                              name='layer_1',
                                                              eps=eps, **kwargs)

            self.layer_2 = stable_nalu.layer.GeneralizedLayer(hidden_size, output_size,
                                                              'linear' if unit_name_2 in stable_nalu.layer.BasicLayer.ACTIVATIONS else unit_name_2,
                                                              writer=self.writer,
                                                              name='layer_2',
                                                              eps=eps, **kwargs)
        self.reset_parameters()
        self.z_1_stored = None

    def parameters(self):
        if hasattr(self.hidden_size, '__iter__'):
            prms = []
            for i in range(len(self.layers)):
                prms.extend(list(self.layers[i].parameters()))
            return iter(prms)
        else:
            return super().parameters()

    def reset_parameters(self):
        if hasattr(self.hidden_size, '__iter__'):
            for l in self.layers:
                l.reset_parameters()
        else:
            self.layer_1.reset_parameters()
            self.layer_2.reset_parameters()

    def regualizer(self):
        if self.nac_mul == 'max-safe':
            return super().regualizer({
                'z': torch.mean(torch.relu(1 - self.z_1_stored))
            })
        else:
            return super().regualizer()

    def forward(self, input):
        if self.actor:
            if len(input.size()) == 1:
                input = input[None]
        if hasattr(self.hidden_size, '__iter__'):
            for l in self.layers:
                input = l(input)
            z_2 = input
        else:

            self.writer.add_summary('x', input)
            z_1 = self.layer_1(input)
            self.z_1_stored = z_1
            self.writer.add_summary('z_1', z_1)

            if self.nac_mul == 'none' or self.nac_mul == 'mnac':
                z_2 = self.layer_2(z_1)
            elif self.nac_mul == 'normal':
                z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1) + self.eps)))
            elif self.nac_mul == 'safe':
                z_2 = torch.exp(self.layer_2(torch.log(torch.abs(z_1 - 1) + 1)))
            elif self.nac_mul == 'max-safe':
                z_2 = torch.exp(self.layer_2(torch.log(torch.relu(z_1 - 1) + 1)))
            else:
                raise ValueError(f'Unsupported nac_mul option ({self.nac_mul})')

            self.writer.add_summary('z_2', z_2)
        if self.actor:
            z_2 = F.softmax(z_2, dim=-1)
        return z_2

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )

    def predict(self, x):
        return self(x).detach().cpu().numpy().flatten()
    
    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().cpu().numpy().flatten()


class A2CAgentRNN(A2CAgentRandom):

    def __init__(self, config, range_train, range_eval, a_net, v_net, LSTM=False, reg=100):
        super(A2CAgentRNN, self).__init__(config, range_train, range_eval)
        self.value_network = v_net
        self.actor_network = a_net
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(),
                                                     lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(),
                                                     lr=config['actor_network']['learning_rate'])
        self.recurrent = LSTM
        self.lambda_reg = reg

    def optimize_model(self, observations, actions, returns, advantages, nb_traj):
        # Optimize value function
        # MSE for the values
        observed = torch.tensor(observations, dtype=torch.float)
        returns_torch = torch.tensor(returns, dtype=torch.float)
        if self.gpu:
            observed = observed.cuda()
            returns_torch = returns_torch.cuda()
        Vs_current = self.value_network(observed).flatten()
        loss_value = F.mse_loss(Vs_current, returns_torch)
        if hasattr(self.value_network, 'regualizer'):
            regu = self.value_network.regualizer()
            loss_value += (regu['W'] + regu['z'] + regu['g'] + regu['W-OOB']) * self.lambda_reg
        self.value_network_optimizer.zero_grad()
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor & Entropy loss
        Policies = self.actor_network(observed)
        Policies_action = torch.stack([x[actions[i]] for i, x in enumerate(Policies)])
        loss_action = - torch.sum(torch.tensor(advantages, dtype=torch.float) * torch.log(Policies_action) +  # actor
                                  0.001 * Policies_action * torch.log(Policies_action)) / nb_traj  # entropy
        if hasattr(self.actor_network, 'regualizer'):
            regu = self.actor_network.regualizer()
            loss_action += (regu['W'] + regu['z'] + regu['g'] + regu['W-OOB']) * self.lambda_reg
        self.actor_network_optimizer.zero_grad()
        loss_action.backward()
        self.actor_network_optimizer.step()
        
    def training_batch(self, epochs, batch_size, disp=True, gpu=False):
        """Perform a training by batch

        Parameters
        ----------
        epochs : int
            Number of epochs
        batch_size : int
            The size of a batch
        """
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + (self.env.observation_space.shape[0] + 1,), dtype=np.float)
        observation = self.env.reset()
        rewards_test = []

        
        if gpu:
            self.actor_network.cuda()
            self.value_network.cuda()
            self.gpu = True
        else:
            self.actor_network.cpu()
            self.value_network.cpu()
            self.gpu = False

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                torch_obs = torch.tensor(observations[i] , dtype=torch.float)
                if self.gpu: torch_obs = torch_obs.cuda()
                actions[i] = self.actor_network.select_action(torch_obs)
        #  values[i] = self.value_network.predict(torch.tensor(observations[i] , dtype=torch.float))
                # step
                observation, reward, done, _ = self.env.step(actions[i])
                if len(observation) == self.env.observation_space.shape[0]:
                    observation = np.insert(observation, len(observation), self.env.mass)
                dones[i] = done
                rewards[i] = reward
                if dones[i]:
                    observation = self.env.reset()
        #    print(self.env.mass)
            
            all_observe = torch.tensor(np.concatenate((observations, observation[None])), dtype=torch.float)
            if self.gpu: all_observe = all_observe.cuda()
            all_values = self.value_network.predict(all_observe)
            values = all_values[:-1]

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = all_values[-1]

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages, max(1, sum(dones)))

            # Test it every 50 epochs
            if epoch % 20 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate(*self.range_eval) for _ in range(50)]))
                if disp: print(f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs -1:
                    if disp: print('Early stopping !')
                    break
                observation = self.env.reset()

        # Plotting
        if disp:
            plt.figure()
            r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Epoch', 'Reward'])
            sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')
        
        if disp: print(f'The training was done over a total of {episode_count} episodes')


class Qfunction(stable_nalu.abstract.ExtendedTorchModule):
    
    UNIT_NAMES = stable_nalu.layer.GeneralizedLayer.UNIT_NAMES

    def __init__(self, unit_names, input_size=100, hidden_size=[32, 16], output_size=2, eps=1e-7, **kwargs):
        super().__init__('network', **kwargs)
        self.unit_names = unit_names
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.eps = eps

        # self.layers = [stable_nalu.layer.GeneralizedLayer(input_size, hidden_size[0],
        #                unit_names[0], writer=self.writer, name='layer_1',
        #                 eps=eps, **kwargs)]
        # for i in range(len(hidden_size)-1):
        #     self.layers.append(
        #         stable_nalu.layer.GeneralizedLayer(hidden_size[i], hidden_size[i+1],
        #                 unit_names[i+1], writer=self.writer, name=f'layer_{i+2}',
        #                 eps=eps, **kwargs)
        #     )
        # self.layers.append(
        #         stable_nalu.layer.GeneralizedLayer(hidden_size[-1], output_size,
        #                 'linear', writer=self.writer, name=f'layer_{len(self.hidden_size)}',
        #                 eps=eps, **kwargs)
        #     )
        self.layers = [nn.Linear(input_size, hidden_size[0])]
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))
        self.reset_parameters()
        
    def parameters(self):
        prms = []
        for i in range(len(self.layers)):
            prms.extend(list(self.layers[i].parameters()))
        return iter(prms)

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()

    def regualizer(self):
        return super().regualizer()

    def forward(self, input):
        if len(input.size()) == 1:
            input = input[None]
        # for l in self.layers:
        #     input = l(input)
        for l in self.layers[:-1]:
            input = F.relu(l(input))
        input = self.layers[-1](input)

        z = input
        return z

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
    
    def predict(self, x):
        return self(x).detach().cpu().numpy().flatten()
    
    def select_action(self, state, eps=0.5):
        # probs = F.softmax(state_action_val, dim=0).flatten()
        # if eps is None:
        #     choice = torch.multinomial(probs, 1).detach().cpu().numpy().flatten()[0]
        # else:
        if np.random.rand() < eps:
            choice = torch.tensor([[random.randrange(2)]], dtype=torch.long)
        else:
            choice = self(state).max(1)[1].view(1, 1) #.numpy().flatten()
        return choice


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args): #state, action, next_state, reward
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QTrainer:

    def __init__(self, config, range_train, range_eval, Q, Qeval, eps_func=None, reg=100, capacity=100000):
        self.config = config
        self.env = RandomWrapper(gym.make(config['env_id']), *range_train)
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']
        self.range_train = range_train
        self.range_eval = range_eval
        self.eps_func = eps_func
        self.memory = ReplayMemory(capacity)

        self.Q = Q
        self.Qeval = Qeval
        # self.Qeval = type(Q)(Q.unit_names, Q.input_size, Q.hidden_size, Q.output_size, Q.eps) # get a new instance
        # self.Qeval.load_state_dict(Q.state_dict()) # copy weights and stuff
        self.Qeval.eval()
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=config['learning_rate'])
        self.began_optim = False

    def _get_state(self, observation, next_observation):
        state = next_observation - observation
        state[-1] = observation[-1] #mass
        return torch.tensor(state, dtype=torch.float).unsqueeze(0)

    def train_model(self, num_episodes, eval_update=10, eval_every=50, batch_size=128):
        rewards = [0]
        for i in tqdm.tqdm_notebook(range(num_episodes)):
            observation = self.env.reset()
            next_observation = observation
            state = self._get_state(observation, next_observation)
            done = False
            reward_i = 0
            # transitions = []
            while not done:
                # state_action_val = self.Q(state)
                if self.eps_func is not None:
                    eps = self.eps_func(i/num_episodes)
                else:
                    eps = None
                action = self.Q.select_action(state, eps=eps)

                # predicted.append(state_action_val[0, action])
                observation = next_observation
                next_observation, reward, done, _ = self.env.step(action.item())
                reward_i += reward
                reward = torch.tensor([reward])
                if not done:
                    next_state = self._get_state(observation, next_observation)
                else:
                    next_state = None
                # transitions.append([state, action, next_state, reward])
                self.memory.push(state, action, next_state, reward)
                # print(state, next_state)
                state = next_state
                # next_state_action_val = self.Q(torch.tensor(next_state, dtype=torch.float)).detach().cpu().numpy()
                # future_val = reward + (self.gamma * np.max(next_state_action_val, axis=1)[0] if not done else 0)
                # future.append(future_val)
                if done:
                    break
            
            # if np.median(np.array(rewards)) < reward_i:
            #     rewards.append(reward_i)
            #     for s, a, n, r in transitions:
            #         self.memory.push(s, a, n, r)
            self.optimize_model(batch_size)
            if i % eval_update == 0:
                # print(list(self.Qeval.parameters())[0][0])
                self.Qeval.load_state_dict(self.Q.state_dict())
            if i % eval_every == 0:
                eval_agent(self, 50, self.range_train[0], self.range_train[1], disp=True)
                # print(f'median train rewards : {np.median(np.array(rewards)):.3f}')
                # print(list(self.Qeval.parameters())[0][0])
                print()
            
    def optimize_model(self, batch_size):
        if len(self.memory) < batch_size:
            return
        if len(self.memory) >= batch_size and not self.began_optim:
            print('optimizing')
            self.began_optim = True
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.Q(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        next_state_values[non_final_mask] = self.Qeval(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # print(np.stack((state_action_values.detach().numpy(),
        #         expected_state_action_values.unsqueeze(1).detach().numpy())))
        # Compute Huber loss
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for p in self.Q.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        # print(f'Loss : {loss:.3f} - Gradient norm : {total_norm:.3f}')
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # self.optimizer.zero_grad()
        # loss = F.mse_loss(torch.stack(predicted), torch.tensor(future))
        # loss.backward()
        # self.optimizer.step()
        # print(f'Loss : {loss:.2f}')

        
        
    def evaluate(self, min_mass, max_mass, render=False):
        if (min_mass, max_mass) == self.range_train: env = self.env
        else: env = RandomWrapper(gym.make(self.config['env_id']), min_mass, max_mass)
        if render: env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)
        observation = env.reset()
        next_observation = observation
        state = self._get_state(observation, next_observation)
        reward_episode = 0
        done = False
        actions = []
        with torch.no_grad():
            while not done:
                # state_action_val = self.Q(state)
                action = self.Q.select_action(state, eps=0)
                actions.append(action[0, 0])
                observation = next_observation
                next_observation, reward, done, _ = env.step(action.item())
                state = self._get_state(observation, next_observation)
                reward_episode += reward
        # print(np.array(actions))
        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
            print(f'masspole : {env.env.env.masspole:.2f}')
        return reward_episode
            
