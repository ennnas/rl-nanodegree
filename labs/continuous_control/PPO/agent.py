import numpy as np

import torch
import torch.optim as optim

from PPO.model import Actor, Critic, Policy
from PPO.utils import Batcher

LR = 1e-3
SGD_EPOCH = 4
DISCOUNT = 0.99
EPSILON = 0.2
BETA = 0.001
TAU = 0.95
BATCH_SIZE = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_gae(next_value, trajectory):
    gae = 0
    values = trajectory['value'] + [next_value]
    returns = []
    n_steps = len(trajectory['reward'])
    # convert arrays to torch tensor
    values = [torch.FloatTensor(v).to(device) for v in values]
    rewards = [torch.FloatTensor(r).unsqueeze(1).to(device) for r in trajectory['reward']]
    masks = [(1-torch.FloatTensor(d).unsqueeze(1).to(device)) for d in trajectory['done']]

    for step in reversed(range(n_steps)):
        delta = rewards[step] + DISCOUNT * values[step + 1] * masks[step] - values[step]
        gae = delta + DISCOUNT * TAU * masks[step] * gae
        returns.insert(0, gae + values[step])
    return torch.cat(returns)


def compute_advantages(returns, values):
    returns = returns.detach()
    values = torch.cat(values).detach()
    advantages = returns - values
    return (advantages - advantages.mean()) / advantages.std()


class Agent(object):
    """
    Implementation of a PPO agent that interacts with and learns from the
    environment
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialize an PPO Agent object.
        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        """
        self.action_size = action_size

        # Policy Network
        actor_net = Actor(state_size, action_size, seed).to(device)
        critic_net = Critic(state_size, seed).to(device)
        self.policy = Policy(action_size, actor_net, critic_net).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)

    def step(self, trajectory, returns, advantages):

        # Learn every UPDATE_EVERY time steps.
        for i in range(SGD_EPOCH):
            self.learn(trajectory, returns, advantages)

    def act(self, states):
        '''
        Returns actions for given states as per current policy.
        :param states: array_like. current states
        '''
        states = torch.from_numpy(states).float().to(device)
        self.policy.eval()
        with torch.no_grad():
            dist, values = self.policy(states)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        self.policy.train()
        return actions, log_probs, values, entropy

    def learn(self, trajectories, returns, advantages):
        '''
        Update policy and value params using given batch of experience tuples.
        :param trajectories: Trajectory object. tuples of (s, a, r, s', done)
        '''
        states = torch.Tensor(trajectories['state'])
        old_probs = torch.cat(trajectories['prob'])
        actions = torch.cat(trajectories['action'])

        # learn in batches
        batcher = Batcher(BATCH_SIZE, [np.arange(states.size(0))])
        batcher.shuffle()
        while not batcher.end():
            batch_indices = batcher.next_batch()[0]
            batch_indices = torch.LongTensor(batch_indices)

            loss = clipped_surrogate(self.policy,
                                          states[batch_indices],
                                          actions[batch_indices],
                                          old_probs[batch_indices],
                                          returns[batch_indices],
                                          advantages[batch_indices])
            # Minimize the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def clipped_surrogate(policy, states, actions, old_probs, returns, advantages, epsilon=EPSILON, beta=BETA):
    dist, values = policy(states)
    entropy_loss = dist.entropy().mean()
    new_probs = dist.log_prob(actions)
    # ratio for clipping. All probabilities used are log probabilities
    # with torch.no_grad():
    ratio = (new_probs - old_probs).exp()

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    actor_loss = torch.min(ratio*advantages, clip*advantages).mean()
    critic_loss = (returns - values).pow(2).mean()

    policy_loss = 0.5 * critic_loss - actor_loss - beta * entropy_loss
    return policy_loss
