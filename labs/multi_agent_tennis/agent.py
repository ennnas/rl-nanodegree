import numpy as np
import random

from model import Actor, Critic
from utils import OUNoise

import torch
from torch.nn import functional as F
import torch.optim as optim

LR_ACTOR = 5e-4           # learning rate of the actor
LR_CRITIC = 5e-4          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay
TAU = 3e-3                # for soft update of target parameters
GRAD_CLIP = 1             # Value at which clip the gradient during backprop


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, agent_id, state_size, action_size, random_seed, num_agents=2):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.seed = random.seed(random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Actor Network (w/ Target Network)
        self.actor = Actor(state_size, action_size, random_seed)
        self.target_actor = Actor(state_size, action_size, random_seed)

        # Critic Network (w/ Target Network)
        self.critic = Critic(num_agents * state_size, num_agents * action_size, random_seed)
        self.target_critic = Critic(num_agents * state_size, num_agents * action_size, random_seed)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

    def act(self, states, epsilon=1.0):
        """Returns actions for given states as per current policy."""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy() + epsilon*self.noise.sample()
        self.actor.train()
        return np.clip(actions, -1, 1)

    def target_act(self, states, epsilon=1.0):
        """Returns actions for given states as per current policy."""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        self.actor.eval()
        actions = self.target_actor(states).cpu().data.numpy() + epsilon*self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, other_next_actions):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- data prep -------------------------------- #
        batch_size = states.shape[0]

        own_next_states = next_states[:, self.agent_id]
        own_states = states[:, self.agent_id]

        other_actions = actions[:, 2:] if self.agent_id == 0 else actions[:, :2]
        own_rewards = rewards[:, self.agent_id].unsqueeze(1)
        own_dones = dones[:, self.agent_id].unsqueeze(1)

        # flatten for input
        states = states.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        next_states = next_states.reshape(batch_size, -1)

        # ---------------------------- update critic ---------------------------- #
        actions_next = self.target_actor(own_next_states)
        if self.agent_id == 0:
            actions_next = torch.cat((actions_next, other_next_actions), dim=1)
        else:
            actions_next = torch.cat((other_next_actions, actions_next), dim=1)

        Q_targets_next = self.target_critic(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = own_rewards + (gamma * Q_targets_next * (1 - own_dones))
        Q_targets = Q_targets.detach()

        # Compute critic loss
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)

        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        pred_actions = self.actor(own_states)

        if self.agent_id == 0:
            actions_pred = torch.cat((pred_actions, other_actions), dim=1)
        else:
            actions_pred = torch.cat((other_actions, pred_actions), dim=1)

        actor_loss = -self.critic(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic, self.target_critic, TAU)
        self.soft_update(self.actor, self.target_actor, TAU)

    def soft_update(self, source, target, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            source (torch.nn.Module): Net whose parameters to copy
            target (torch.nn.Module): Net to copy parameters to
            tau (float, 0 < x <= 1): Weight factor for update, set to 1 to make hard copy
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
