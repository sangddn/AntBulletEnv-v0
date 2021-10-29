import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def NetworkGenerator(sizes):

    layers = []
    for i in range(len(sizes)-1):
        if i < len(sizes) - 2:
            activation = nn.ReLU
        else:
            activation = nn.Identity
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation()]

    return nn.Sequential(*layers)


class Actor(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        log_std = -0.5 * np.ones(n_outputs)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # Network
        self.actor_net = NetworkGenerator([n_inputs] + [64, 64] + [n_outputs])

    # Probability dist.
    def prob(self, state):
        mu = self.actor_net(state)
        std = torch.exp(self.log_std)
        prob = Normal(mu, std)
        return prob

    # Log probability dist.
    def logp(self, prob, action):
        return prob.log_prob(action).sum(-1)

    def step(self, state):
        with torch.no_grad():
            prob = self.prob(state)
            action = prob.sample()
            logp_a = self.logp(prob, action)
        return action.numpy(), logp_a.numpy()

    def forward(self, state, action=None):
        # Probability
        prob = self.prob(state)

        # Log probability
        logp = None
        if action is not None:
            logp = self.logp(prob, action)

        return prob, logp


class Target(nn.Module):

    def __init__(self, n_inputs):
        super().__init__()
        # Network
        self.target_net = NetworkGenerator([n_inputs] + [64, 64] + [1])

    def forward(self, state):
        p_c = self.target_net(state)
        p_c = torch.squeeze(p_c, -1)

        return p_c


class AC(nn.Module):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        # Actor network
        self.actor_net = Actor(n_inputs, n_outputs)
        # Critic network
        self.target_net = Target(n_inputs)

    def step(self, state):
        with torch.no_grad():
            prob = self.actor_net.prob(state)
            action = prob.sample()
            logp_a = self.actor_net.logp(prob, action)
            target = self.target_net(state)
        return action.numpy(), target.numpy(), logp_a.numpy()

    def act(self, state):
        return self.step(state)[0]
