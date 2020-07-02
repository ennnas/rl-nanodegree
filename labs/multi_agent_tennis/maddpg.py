import logging
from agent import Agent
from utils import ReplayBuffer, permute
import numpy as np

BUFFER_SIZE = int(1e5)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
UPDATE_EVERY = 1          # how often to update the network
EPSILON_DECAY = 0.9       # decay the noise factor

logging.basicConfig(level=logging.INFO)


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

#    def update(self, samples, agent_number):
#        """update the critics and actors of all the agents """
#        # A batch of transitions is sampled from the replay buffer.
#        states, actions, rewards, next_states, dones = map(permute, samples)
#        batch_size = states.shape[1]
#        logging.info(f"\n Batch size: {batch_size}")
#
#        # Critic models learns Q-values for combined states and actions from all the agents.
#        # While actors only relies on local information. *_full is used as global information for the critic
#        states_full = permute(states).reshape(batch_size, -1)
#        next_states_full = permute(next_states).reshape(batch_size, -1)
#        actions_full = permute(actions).reshape(batch_size, -1)
#
#        agent = self.agents[agent_number]
#        agent.critic_optimizer.zero_grad()
#
#        # ---------------------------- Update online critic model -------------------------------- #
#        # critic loss = batch mean of (y- Q(s,a) from target network)^2
#        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
#
#        # Compute actions for the next states with the target actor model
#        target_actions = self.target_act(next_states)
#        target_actions = permute(torch.Tensor(target_actions)).reshape(batch_size, -1)
#
#        logging.info(f"next_state_full: {next_states_full.shape}")
#        logging.info(f"target_action: {target_actions.shape}")
#
#        with torch.no_grad():
#            # Compute Q-values for the next states and actions with the target critic model
#            Q_next = agent.target_critic(next_states_full, target_actions)
#
#        # Compute target Q-values for the current states and actions using the Bellman equation
#        y = rewards[agent_number].unsqueeze(1) + GAMMA * Q_next * (1 - dones[agent_number].unsqueeze(1))
#        logging.info(actions.shape)
#        # Compute Q-values for the current states and actions with the online critic model
#        q = agent.critic(states_full, actions_full)
#
#        # Use the target and online Q-values to compute the loss
#        critic_loss = F.mse_loss(q, y.detach())
#        critic_loss.backward()
#        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), GRAD_CLIP)
#        # Minimize the loss for the online critic model, that's why we have y.detach()
#        agent.critic_optimizer.step()
#
#        # ------------------ Update online actor network using policy gradient ------------------- #
#        agent.actor_optimizer.zero_grad()
#
#        # Compute actions for current states with the online actor model
#        current_actions = [self.agents[i].actor(state) if i == agent_number \
#                       else self.agents[i].actor(state).detach()
#                   for i, state in enumerate(states)]
#
#        # combine all the actions and observations for input to critic
#        current_actions = torch.cat(current_actions, dim=1)
#        logging.info(current_actions.shape)
#
#        # Compute Q-values with the online critic model
#        actor_loss = -agent.critic(states_full, current_actions).mean()
#        actor_loss.backward()
#        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), GRAD_CLIP)
#        agent.actor_optimizer.step()
#
#        #self.soft_update(agent.target_actor, agent.actor, TAU)
#        #self.soft_update(agent.target_critic, agent.critic, TAU)

