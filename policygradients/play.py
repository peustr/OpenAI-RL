import sys

import gym

from agent import REINFORCEAgent


try:
    env_name = sys.argv[1]
except IndexError:
    env_name = "CartPole-v0"

try:
    model_filename = sys.argv[2]
except IndexError:
    model_filename = "CartPole-v0_5000e.h5"

try:
    num_episodes = int(sys.argv[3])
except IndexError:
    num_episodes = 10

print("Playing agent {} for {} episodes.".format(model_filename, num_episodes))

env = gym.make(env_name)
agent = REINFORCEAgent(env, model_filename=model_filename)

for i_episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
    print("Episode {} reward: {}".format(i_episode + 1, total_reward))

env.close()
