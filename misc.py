'''

Miscellaneous functions, classes and hyperparameters

'''

import numpy as np
import torch
import scipy.signal

from utils.mpi_tools import mpi_statistics_scalar


# Hyperparameters
class HyperParams:

    # Data-saving frequency (used for visualization)
    save_freq = 10

    # Number of epochs and steps per each
    epochs = 2000
    steps_per_epoch = 1000

    # Number of steps when gradient-ascending
    n_iters_target = 80
    n_iters_actor = 80

    # Discount factors
    lam = 0.97
    gamma = 0.99

    # Learning rates (alpha and beta)
    actor_lr = 3e-4
    critic_lr = 1e-3

    # PPO-related parameters
    clip_alpha = 0.2  # for clipped advantage function
    target_kl = 0.01  # for KL Divergence


# Memory class used to store experience for PPO and PGB
class Memory:

    def __init__(self, n_inputs, n_outputs, size):

        self.states = np.zeros((size, n_inputs), dtype=np.float32)
        self.actions = np.zeros((size, n_outputs), dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)

        # Pointers
        self.index, self.starts_here = 0, 0

    def save(self, state, action, reward, value, logp):

        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.values[self.index] = value
        self.logp[self.index] = logp

        self.index += 1

    def finish(self, Vs_T=0):

        path = slice(self.starts_here, self.index)
        rewards = np.append(self.rewards[path], Vs_T)
        values = np.append(self.values[path], Vs_T)

        # GAE-Lambda advantage
        deltas = rewards[:-1] + HyperParams.gamma * values[1:] - values[:-1]
        self.advantages[path] = discount_cumsum(deltas, HyperParams.gamma * HyperParams.lam)

        # Target values
        self.returns[path] = discount_cumsum(rewards, HyperParams.gamma)[:-1]

        self.starts_here = self.index

    def recall(self):

        self.index, self.starts_here = 0, 0

        # Normalization
        adv_mean, adv_std = mpi_statistics_scalar(self.advantages)
        self.advantages = (self.advantages - adv_mean) / adv_std

        dat = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            logp=self.logp
        )

        out = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in dat.items()}

        return out


# Memory class used to store exp for VPG
class VPGMemory:
    def __init__(self, n_inputs, n_outputs, size):

        self.states = np.zeros((size, n_inputs), dtype=np.float32)
        self.actions = np.zeros((size, n_outputs), dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)

        # Pointers
        self.index, self.starts_here = 0, 0

    def save(self, state, action, reward, logp):

        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.logp[self.index] = logp

        self.index += 1

    def finish(self):

        path = slice(self.starts_here, self.index)
        rewards = np.append(self.rewards[path], 0)

        # Target values
        self.returns[path] = discount_cumsum(rewards, HyperParams.gamma)[:-1]

        self.starts_here = self.index

    def recall(self):

        self.index, self.starts_here = 0, 0

        dat = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            logp=self.logp
        )

        out = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in dat.items()}

        return out


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
