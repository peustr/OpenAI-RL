import gym
import numpy as np


def greedy_agent_experiment(env, num_episodes=500):
    np.random.seed(0)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    r_table = np.zeros((state_dim, action_dim))
    for i_episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            # If known rewards are 0, act randomly.
            if np.sum(r_table[s, :]) == 0:
                a = np.random.randint(action_dim)
            else:
                # Select the action with the highest cumulative reward.
                a = np.argmax(r_table[s, :])
            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table


def q_learning_experiment(env, num_episodes=500):
    np.random.seed(0)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    q_table = np.zeros((state_dim, action_dim))
    gamma = 0.9
    lr = 0.5
    for i_episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
        	# If known rewards are 0, act randomly.
            if np.sum(q_table[s,:]) == 0:
                a = np.random.randint(action_dim)
            else:
                # Select the action that yields the highest Q value.
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (gamma * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table

def q_learning_epsilon_greedy_experiment(env, num_episodes=500):
    np.random.seed(0)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    q_table = np.zeros((state_dim, action_dim))
    gamma = 0.9
    lr = 0.5
    epsilon = 0.9
    for i_episode in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
        	# If known rewards are 0, or with probability 1 - epsilon, act randomly.
            if np.sum(q_table[s,:]) == 0 or np.random.rand() > epsilon:
                a = np.random.randint(action_dim)
            else:
                # Select the action that yields the highest Q value.
                a = np.argmax(q_table[s, :])
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (gamma * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table


env = gym.make("NChain-v0")

tbl_greedy = greedy_agent_experiment(env)
print("Greedy agent:")
print(tbl_greedy)

tbl_q_learning = q_learning_experiment(env)
print("Q-Learning agent:")
print(tbl_q_learning)

tbl_q_learning_eps_greedy = q_learning_epsilon_greedy_experiment(env)
print("Q-Learning epsilon greedy agent:")
print(tbl_q_learning_eps_greedy)
