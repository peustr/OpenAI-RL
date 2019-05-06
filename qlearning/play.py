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
    model_filename = sys.argv[2]
except IndexError:
    model_filename = "Boxing-ram-v0_10000e.h5"

env = gym.make(env_name)
agent = QLearningAgent(env, model_filename=model_filename)

state = normalize_state(env.reset())
done = False
while not done:
    env.render()
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    state = normalize_state(state)

env.close()
