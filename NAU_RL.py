"""
Helper functions for RL cartpole simulation
"""

import math
import base64
import argparse
from pathlib import Path

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

    def __init__(self, unit_names, input_size=100, hidden_size=[32, 16], output_size=1, eps=1e-7, **kwargs):
        super().__init__('network', **kwargs)
        self.unit_names = unit_names
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.eps = eps

        self.layers = [stable_nalu.layer.GeneralizedLayer(input_size, hidden_size[0],
                       unit_names[0], writer=self.writer, name='layer_1',
                        eps=eps, **kwargs)]
        for i in range(len(hidden_size)-1):
            self.layers.append(
                stable_nalu.layer.GeneralizedLayer(hidden_size[i], hidden_size[i+1],
                        unit_names[i+1], writer=self.writer, name=f'layer_{i+2}',
                        eps=eps, **kwargs)
            )
        self.layers.append(
                stable_nalu.layer.GeneralizedLayer(hidden_size[-1], output_size,
                        'linear', writer=self.writer, name=f'layer_{len(self.hidden_size)}',
                        eps=eps, **kwargs)
            )
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
        for l in self.layers:
            input = l(input)
        z = input
        return z

    def extra_repr(self):
        return 'unit_name={}, input_size={}'.format(
            self.unit_name, self.input_size
        )
    
    def predict(self, x):
        return self(x).detach().cpu().numpy().flatten()
    
    def select_action(self, state_action_val):
        probs = F.softmax(state_action_val, dim=0).flatten()
        choice = torch.multinomial(probs, 1)
        return choice.detach().cpu().numpy().flatten()[0]


class QTrainer:

    def __init__(self, config, range_train, range_eval, Q, reg=100):
        self.config = config
        self.env = RandomWrapper(gym.make(config['env_id']), *range_train)
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']
        self.range_train = range_train
        self.range_eval = range_eval

        self.Q = Q
        self.optimizer = optim.RMSprop(self.Q.parameters(), lr=config['learning_rate'])

    # def _state_action(self, state):
    #     t_state = torch.tensor(state, dtype=torch.float)
    #     return torch.stack([torch.cat([t_state, torch.tensor([0.0])]),
    #                         torch.cat([t_state, torch.tensor([1.0])])])
    
    def optimize_model(self, num_epochs, batch_size):
        for j in tqdm.tqdm_notebook(range(num_epochs)):
            observation = self.env.reset()
            predicted, future = [], []
            for i in range(batch_size):
                state_action_val = self.Q(torch.tensor(observation, dtype=torch.float))
                action = self.Q.select_action(state_action_val)
                predicted.append(state_action_val[0, action])

                observation, reward, done, _ = self.env.step(action)
                next_state = self.Q(torch.tensor(observation, dtype=torch.float)).detach().cpu().numpy()
                future_val = reward + (self.gamma * np.max(next_state, axis=1)[0] if not done else 0)
                future.append(future_val)
                if done:
                    observation = self.env.reset()

            self.optimizer.zero_grad()
            loss = F.mse_loss(torch.stack(predicted), torch.tensor(future))
            loss.backward()
            self.optimizer.step()
            print(f'Loss : {loss:.2f}')

            if j % 10 == 0:
                eval_agent(self, 50, self.range_train[0], self.range_train[1], disp=True)
        
    def evaluate(self, min_mass, max_mass, render=False):
        if (min_mass, max_mass) == self.range_train: env = self.env
        else: env = RandomWrapper(gym.make(self.config['env_id']), min_mass, max_mass)
        if render: env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)
        observation = env.reset()
        reward_episode = 0
        done = False
        with torch.no_grad():
            while not done:
                state_action_val = self.Q(torch.tensor(observation, dtype=torch.float))
                action = self.Q.select_action(state_action_val)
                observation, reward, done, _ = env.step(action)
                reward_episode += reward

        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
            print(f'masspole : {env.env.env.masspole:.2f}')
        return reward_episode
            
