import gym
import numpy as np
import sys
from agent import QLearningAgent
from datetime import datetime


try:
    env_name = sys.argv[1]
except IndexError:
    env_name = "Boxing-ram-v0"

try:
    num_episodes = int(sys.argv[2])
except IndexError:
    num_episodes = 2000


def norm_state(state):
    return (np.array(state) / 255).reshape(1, len(state))


env = gym.make(env_name)
agent = QLearningAgent(env)
for i_episode in range(num_episodes):
    ts_start = datetime.now()
    state = norm_state(env.reset())
    done = False
    total_reward = 0
    while not done:
        env.render()  # Comment out for faster training.
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = norm_state(next_state)
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
