# Taxi Problem

### Getting Started

Read the description of the environment in subsection 3.1 of [this paper](https://arxiv.org/pdf/cs/9905014.pdf).  You can verify that the description in the paper matches the OpenAI Gym environment by peeking at the code [here](https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py).


### Instructions

The repository contains three files:
- `agent.py`: Develop your reinforcement learning agent here.  This is the only file that you should modify.
- `monitor.py`: The `interact` function tests how well your agent learns from interaction with the environment.
- `main.py`: Run this file in the terminal to check the performance of your agent.

Navigate to the current folder and run the following command in the terminal:
```
python main.py --num-episodes 100 --alpha 0.1
```

When you run `main.py`, the agent specified in `agent.py` interacts with the environment for N episodes. 
The details of the interaction are specified in `monitor.py`, which returns two variables: `avg_rewards` and `best_avg_reward`.
- `avg_rewards` is a deque where `avg_rewards[i]` is the average (undiscounted) return collected by the agent from episodes `i+1` to episode `i+100`, inclusive.  So, for instance, `avg_rewards[0]` is the average return collected by the agent over the first 100 episodes.
- `best_avg_reward` is the largest entry in `avg_rewards`.  This is the final score used when determining how well the agent performed in the task.

OpenAI Gym [defines "solving"](https://gym.openai.com/envs/Taxi-v1/) this task as getting average return of 9.7 over 100 consecutive trials.  
