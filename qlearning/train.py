import sys
from datetime import datetime

import gym

from agent import QLearningAgent
from utils import normalize_state


try:
    env_name = sys.argv[1]
except IndexError:
    env_name = "Boxing-ram-v0"

try:
    num_episodes = int(sys.argv[2])
except IndexError:
    num_episodes = 2000

env = gym.make(env_name)
agent = QLearningAgent(env)

for i_episode in range(num_episodes):
    # For timing every episode.
    ts_start = datetime.now()
    # For tracking accumulative reward.
    total_reward = 0

    lives = env.env.ale.lives()

    state = normalize_state(env.reset())
    done = False
    while not done:
        # Comment out env.render() for faster training.
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = normalize_state(next_state)
        # If a life is lost, pass terminal state
        if info["ale.lives"] < lives:
            lives = info["ale.lives"]
            done = True
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    ts_end = datetime.now()
    episode_interval = (ts_end - ts_start).total_seconds()
    print("Episode {} ended in {} seconds. Total reward: {}".format(i_episode + 1, episode_interval, total_reward))

    agent.train()

model_filename = "{}_{}e.h5".format(env_name, num_episodes)
agent.save_model(model_filename)

env.close()
