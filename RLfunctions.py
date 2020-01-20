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

import gym
from gym.wrappers import Monitor
from pprint import pprint
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


def show_video(directory):
    html = []
    for mp4 in Path(directory).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append('''<video alt="{}" autoplay 
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>'''.format(mp4, video_b64.decode('ascii')))
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


class ValueNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def predict(self, x):
        return self(x).detach().numpy()


class ActorNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out), dim=-1)
        return out

    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()


class A2CAgent:

    def __init__(self, config):
        self.config = config
        self.env = gym.make(config['env_id'])
        make_seed(config['seed'])
        self.env.seed(config['seed'])
        self.monitor_env = Monitor(self.env, "./gym-results", force=True, video_callable=lambda episode: True)
        self.gamma = config['gamma']

        # Our two networks
        self.value_network = ValueNetwork(self.env.observation_space.shape[0], 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0], 16, self.env.action_space.n)

        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(),
                                                     lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(),
                                                     lr=config['actor_network']['learning_rate'])

    def _compute_returns(self, rewards):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            The array of rewards of one episode

        Returns
        -------
        array
            The cumulative discounted rewards at each time step
            
        Example
        -------
        for rewards=[1, 2, 3] this method outputs [1 + 2 * gamma + 3 * gamma**2, 2 + 3 * gamma, 3] 
        """
        n_r = len(rewards)
        array = np.zeros(n_r)
        gamma_arr = np.full(n_r, self.gamma)
        exponents = np.arange(n_r)[::-1]
        for i, r in enumerate(rewards):
            array[:i + 1] += r * np.full(i + 1, self.gamma) ** exponents[n_r - i - 1:]
        return array

    # Hint: use it during training_batch
    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network
        
        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        stops = np.argwhere(dones).flatten()
        stops = np.r_[stops, len(dones)].astype(int)
        if len(stops) == 1:
            returns = self._compute_returns(np.r_[rewards, next_value])[:-1]
        else:
            returns = np.zeros(rewards.shape)
            start = 0
            for j, stop in enumerate(stops):
                if j == len(stops) - 1:
                    returns[start:stop] = self._compute_returns(np.r_[rewards[start:stop], next_value])[:-1]
                else:
                    returns[start:stop] = self._compute_returns(rewards[start:stop])
                start = stop
                # advantages = returns + self.gamma * np.r_[values[1:], next_value] - values
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
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
        observations = np.empty((batch_size,) + self.env.observation_space.shape, dtype=np.float)
        observation = self.env.reset()
        rewards_test = []

        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                actions[i] = self.actor_network.select_action(torch.tensor(observations[i], dtype=torch.float))
                values[i] = self.value_network.predict(torch.tensor(observations[i], dtype=torch.float))
                # step
                observation, reward, done, _ = self.env.step(actions[i])
                dones[i] = done
                rewards[i] = reward
                if dones[i]:
                    observation = self.env.reset()

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network.predict(torch.tensor(observations[-1], dtype=torch.float))

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages, max(1, sum(dones)))

            # Test it every 50 epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(
                    f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()

        # Plotting
        r = pd.DataFrame(
            (itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))),
            columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd');

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages, nb_traj):
        # Optimize value function
        # MSE for the values
        Vs_current = self.value_network(torch.tensor(observations, dtype=torch.float)).flatten()
        loss_value = F.mse_loss(Vs_current, torch.tensor(returns, dtype=torch.float))
        self.value_network_optimizer.zero_grad()
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor & Entropy loss
        Policies = self.actor_network(torch.tensor(observations, dtype=torch.float))
        Policies_action = torch.stack([x[actions[i]] for i, x in enumerate(Policies)])
        loss_action = - torch.sum(torch.tensor(advantages, dtype=torch.float) * torch.log(Policies_action) +  # actor
                                  0.001 * Policies_action * torch.log(Policies_action)) / nb_traj  # entropy
        self.actor_network_optimizer.zero_grad()
        loss_action.backward()
        self.actor_network_optimizer.step()

    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False

        while not done:
            policy = self.actor_network(observation)
            action = torch.multinomial(policy, 1)
            observation, reward, done, info = env.step(int(action))
            observation = torch.tensor(observation, dtype=torch.float)
            reward_episode += reward

        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
        return reward_episode


class RandomWrapper(gym.Wrapper):
    def __init__(self, env, min_mass, max_mass):
        super().__init__(env)
        self.env = env
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.mass = np.random.uniform(low=self.min_mass, high=self.max_mass)
        self.env.env.masspole = self.mass

    def reset(self):
        #         return self.env.reset()
        observation = self.env.reset()
        self.mass = np.random.uniform(self.min_mass, self.max_mass)
        self.env.env.masspole = self.mass
        observation = np.insert(observation, len(observation), self.mass)
        return observation

    def step(self, action):
        observ, reward, done, arg = super().step(action)
        observ = np.insert(observ, len(observ), self.mass)
        return observ, reward, done, arg


class A2CAgentRandom(A2CAgent):

    def __init__(self, config, range_train, range_eval):
        super(A2CAgentRandom, self).__init__(config)
        self.env = RandomWrapper(self.env, *range_train)
        self.range_train = range_train
        self.range_eval = range_eval

        # Our two networks
        self.value_network = ValueNetwork(self.env.observation_space.shape[0] + 1, 16, 1)
        self.actor_network = ActorNetwork(self.env.observation_space.shape[0] + 1, 16, self.env.action_space.n)

        # Their optimizers
        self.value_network_optimizer = optim.RMSprop(self.value_network.parameters(),
                                                     lr=config['value_network']['learning_rate'])
        self.actor_network_optimizer = optim.RMSprop(self.actor_network.parameters(),
                                                     lr=config['actor_network']['learning_rate'])

    def training_batch(self, epochs, batch_size):
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


        for epoch in range(epochs):
            # Lets collect one batch
            for i in range(batch_size):
                observations[i] = observation
                actions[i] = self.actor_network.select_action(torch.tensor(observations[i], dtype=torch.float))
                values[i] = self.value_network.predict(torch.tensor(observations[i], dtype=torch.float))
                # step
                observation, reward, done, _ = self.env.step(actions[i])
                if len(observation) == self.env.observation_space.shape[0]:
                    observation = np.insert(observation, len(observation), self.env.mass)
                dones[i] = done
                rewards[i] = reward
                if dones[i]:
                    observation = self.env.reset()
            #                 print(self.env.mass)

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network.predict(torch.tensor(observations[-1], dtype=torch.float))

            # Update episode_count
            episode_count += sum(dones)

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages, max(1, sum(dones)))

            # Test it every 50 epochs
            if epoch % 50 == 0 or epoch == epochs - 1:
                rewards_test.append(np.array([self.evaluate(*self.range_eval) for _ in range(50)]))
                print(
                    f'Epoch {epoch}/{epochs}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

                # Early stopping
                if rewards_test[-1].mean() > 490 and epoch != epochs - 1:
                    print('Early stopping !')
                    break
                observation = self.env.reset()

        # Plotting
        plt.figure()
        r = pd.DataFrame(
            (itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))),
            columns=['Epoch', 'Reward'])
        sns.lineplot(x="Epoch", y="Reward", data=r, ci='sd')

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages, nb_traj):
        # Optimize value function
        # MSE for the values
        Vs_current = self.value_network(torch.tensor(observations, dtype=torch.float)).flatten()
        loss_value = F.mse_loss(Vs_current, torch.tensor(returns, dtype=torch.float))
        self.value_network_optimizer.zero_grad()
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor & Entropy loss
        Policies = self.actor_network(torch.tensor(observations, dtype=torch.float))
        Policies_action = torch.stack([x[actions[i]] for i, x in enumerate(Policies)])
        loss_action = - torch.sum(torch.tensor(advantages, dtype=torch.float) * torch.log(Policies_action) +  # actor
                                  0.001 * Policies_action * torch.log(Policies_action)) / nb_traj  # entropy
        self.actor_network_optimizer.zero_grad()
        loss_action.backward()
        self.actor_network_optimizer.step()

    def evaluate(self, min_mass, max_mass, render=False):
        env = RandomWrapper(self.env.env, min_mass, max_mass)
        if render:
            env = Monitor(env, "./gym-results", force=True, video_callable=lambda episode: True)
        observation = env.reset()
        observation = torch.tensor(observation, dtype=torch.float)
        reward_episode = 0
        done = False
        # print(env.mass)
        move_to_gpu = False
        if hasattr(self, 'gpu'):
            if self.gpu:
                move_to_gpu = True
        with torch.no_grad():
            while not done:
                if move_to_gpu:
                    observation = observation.cuda()
                policy = self.actor_network(observation)
                action = torch.multinomial(policy, 1)
                observation, reward, done, info = env.step(int(action))
                if len(observation) == self.env.observation_space.shape[0]:
                    observation = np.insert(observation, len(observation), self.env.mass)
                observation = torch.tensor(observation, dtype=torch.float)
                reward_episode += reward

        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
            print(f'masspole : {env.env.env.masspole:.2f}')
        return reward_episode


def eval_agent(agent, n_eval, min_mass, max_mass, disp=False):
    rws = []
    for _ in tqdm.tqdm_notebook(range(n_eval), disable=True):
        r = agent.evaluate(min_mass, max_mass)
        rws.append(r)
    rws = np.array(rws)
    if disp: print(f'Mean success : {rws.mean():.2f} +- {rws.std():.2f}')
    return rws
