import gym
import sys
from agent import QLearningAgent


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

state_dim = env.observation_space.shape[0]
for i_episode in range(num_episodes):
    print("Episode:", i_episode + 1)
    state = env.reset().reshape(1, state_dim)
    done = False
    while not done:
        env.render()  # Comment out for faster training.
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, state_dim)
        agent.remember(state, action, reward, next_state)
        state = next_state
    agent.train()

model_filename = "{}_{}e".format(env_name, num_episodes)
agent.save_model(model_filename)

env.close()
