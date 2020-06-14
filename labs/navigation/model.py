import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1_value = nn.Linear(in_features=state_size, out_features=32)
        self.fc2_value = nn.Linear(in_features=32, out_features=1)

        self.fc1_adv = nn.Linear(in_features=state_size, out_features=64)
        self.fc2_adv = nn.Linear(in_features=64, out_features=action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x_value = F.relu(self.fc1_value(state))
        v = self.fc2_value(x_value)

        x_adv = F.relu(self.fc1_adv(state))
        adv = self.fc2_adv(x_adv)
        return v + adv - adv.mean()
