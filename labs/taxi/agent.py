import numpy as np
from collections import defaultdict


class Agent:
    def __init__(
        self,
        n_actions: int = 6,
        alpha: float = 0.1,
        gamma: float = 1.0,
        min_eps: float = 0.1,
        eps_decay: float = 0.999,
    ):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.min_eps = min_eps
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
        self.eps = 1.0

    def __repr__(self) -> str:
        return f"alpha: {self.alpha} gamma: {self.gamma}, min_eps: {self.min_eps} eps_decay: {self.eps_decay}"

    def select_action(self, state: int) -> int:
        """ Given the state, select an action using an epsilon greedy approach.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() > self.eps:
            return int(np.argmax(self.Q[state]))
        else:
            return np.random.choice(np.arange(self.n_actions))

    def step(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] += self.alpha * (
            reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action]
        )
        if done:
            self.eps = max(self.eps_decay * self.eps, self.min_eps)
