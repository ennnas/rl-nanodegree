from collections import deque, namedtuple

import copy
import numpy as np
import random
import torch


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience", field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add new experiences to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([[e.state] for e in experiences if e is not None]))
            .float()
        )
        actions = (
            torch.from_numpy(np.vstack([[e.action] for e in experiences if e is not None]))
            .float()
        )
        rewards = (
            torch.from_numpy(np.vstack([[e.reward] for e in experiences if e is not None]))
            .float()
        )
        next_states = (
            torch.from_numpy(np.vstack([[e.next_state] for e in experiences if e is not None]))
            .float()
        )
        dones = (
            torch.from_numpy(
                np.vstack([[e.done] for e in experiences if e is not None]).astype(np.uint8)
            )
            .float()
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def permute(tensor):
    """ Transpose a tensor with shape [x, y, z] into [y, x, z] """
    new_dims = [1, 0] + list(range(2, tensor.dim()))
    return tensor.permute(new_dims)
