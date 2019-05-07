import random

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense


class DQNAgent(object):
    def __init__(self, env, epsilon=0.95, gamma=0.9, memory_size=1000, model_filename=None):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        if model_filename is None:
            self.model = build_model(self.state_dim, self.action_dim)
        else:
            self.model = load_model(model_filename)
        self.memory = []
        self.memory_size = memory_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]

    def train(self, batch_size=32):
        X = []
        y = []
        for state, action, reward, next_state, done in self.memory:
            if done:
                expected_reward = reward
            else:
                expected_reward = reward + self.gamma * np.max(self.model.predict(next_state)[0])
            predicted_q_values = self.model.predict(state)[0]
            predicted_q_values[action] = expected_reward
            X.append(state[0])
            y.append(predicted_q_values)
        self.model.fit(np.array(X), np.array(y), batch_size=batch_size, epochs=1, verbose=0)

    def act(self, state):
        if np.random.rand() > self.epsilon:
            return self.env.action_space.sample()
        q_values = self.model.predict(state)[0]
        return np.argmax(q_values)

    def save_model(self, filename):
        self.model.save(filename)


def build_model(state_dim, action_dim):
    model = Sequential()
    model.add(Dense(state_dim, input_dim=state_dim, activation="relu"))
    model.add(Dense(state_dim, activation="relu"))
    model.add(Dense(action_dim, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    return model
