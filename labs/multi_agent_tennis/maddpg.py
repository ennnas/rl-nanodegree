from agent import Agent
from utils import ReplayBuffer
import numpy as np

BUFFER_SIZE = int(1e5)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
UPDATE_EVERY = 1          # how often to update the network
EPSILON_DECAY = 0.9       # decay the noise factor


class MADDPG:
    def __init__(self, state_size, action_size, random_seed, n_agents=2):
        super().__init__()

        self.agents = [Agent(agent_id, state_size, action_size, random_seed) for agent_id in range(n_agents)]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.n_agents = n_agents
        self.timestep = 0
        self.epsilon = 1.0

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, agents_states):
        """get actions from all actor agents in the MADDPG object"""
        actions = [agent.act(state, self.epsilon) for agent, state in zip(self.agents, agents_states)]
        self.epsilon *= EPSILON_DECAY
        return np.array(actions).reshape(-1)

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        # update once after every episode_per_update
        self.timestep = (self.timestep + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.timestep == 0:
            samples = [self.memory.sample() for _ in range(self.n_agents)]
            self.learn(samples, GAMMA)

    def learn(self, samples, gamma):
        for agent_id, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = samples[agent_id]
            next_state = next_states[:, agent_id, :]
            other_agent = self.agents[1] if agent_id == 0 else self.agents[0]
            other_next_actions = other_agent.target_actor(next_state).detach()
            agent.learn(samples[agent_id], gamma, other_next_actions)
