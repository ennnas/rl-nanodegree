import argparse
from agent import Agent
from monitor import interact
import gym


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an agent on the Taxi-v3 OpenAI Gym env."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20000,
        help="the number of episodes to train the agent for",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="the alpha value used by the agent to update the Q-table",
    )
    parser.add_argument(
        "--eps-decay",
        type=float,
        default=0.9999,
        help="the epsilon decay used by the agent",
    )
    parser.add_argument(
        "--eps-min",
        type=float,
        default=0.005,
        help="the minimum value of epsilon used by the agent",
    )

    args = parser.parse_args()

    env = gym.make("Taxi-v3")
    agent = Agent(alpha=args.alpha, min_eps=args.eps_min, eps_decay=args.eps_decay)
    print(agent)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=args.num_episodes)
