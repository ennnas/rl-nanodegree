# Lunar Lander Problem

### Getting Started

This lab makes uses of the [LunarLander-v2](http://gym.openai.com/envs/LunarLander-v2/) OpenAI gym environment.
Two different agents have been tested
- [Vanilla DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf): Inspired by the work of DeepMind on the Atari 2600 games
- [Dueling DQN](https://arxiv.org/pdf/1511.06581.pdf): A variation of the Vanilla DQN approach proposed by DeepMind that makes use of two separate branch to measure the value of states and the action advantage

The two general architectures are shown in the image below

![DQN architectures](https://i.stack.imgur.com/XnOaj.png)

### Instructions

The folder contains several files:
- `Deep_Q_Network.ipynb`: The notebook used to train and evaluate the agents.  This is the only file that you need to run.
- `dqn_agent.py`: The `agent` class is a wrapper around the RL agent and offers 3 endpoints: step, act and learn.
- `model.py`: This file contains the PyTorch definitions of the Vanilla and Dueling DQN architectures.
- `*.pth`: The checkpoint of the trained agent, i.e. the `qnetwork_local` instance of the agent. 

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/LunarLander-v2/) this task as getting average return of 200 over 100 consecutive trials.  
