import sys
from datetime import datetime

import gym

from agent import REINFORCEAgent


try:
    env_name = sys.argv[1]
except IndexError:
    env_name = "CartPole-v0"

try:
    num_episodes = int(sys.argv[2])
except IndexError:
    num_episodes = 2000

env = gym.make(env_name)
agent = REINFORCEAgent(env)

for i_episode in range(num_episodes):
    # For timing every episode.
    ts_start = datetime.now()
    # For tracking accumulative reward.
    total_reward = 0

    state = env.reset()
    done = False
    states, actions, rewards = [], [], []
    while not done:
        # Comment out env.render() for faster training.
        # env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
    agent.train(states, actions, rewards)

    ts_end = datetime.now()
    episode_interval = (ts_end - ts_start).total_seconds()
    print("Episode {} ended in {} seconds. Total reward: {}".format(i_episode + 1, episode_interval, total_reward))

model_filename = "{}_{}e.h5".format(env_name, num_episodes)
agent.save_model(model_filename)

env.close()
