import gym
from qlearning import QLearningAgent

env = gym.make("Boxing-ram-v0")
agent = QLearningAgent(env)

state_dim = env.observation_space.shape[0]
for i_episode in range(1000):
    print("Episode:", i_episode + 1)
    state = env.reset().reshape(1, state_dim)
    done = False
    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, state_dim)
        agent.remember(state, action, reward, next_state)
        state = next_state
    agent.train()

agent.save_model("boxing_1000e.h5")

env.close()
