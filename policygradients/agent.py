import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical


class REINFORCEAgent(object):
    def __init__(self, env, gamma=0.99, model_filename=None):
        self.env = env
        self.gamma = gamma
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        if model_filename is None:
            self.model = build_model(self.state_dim, self.action_dim)
        else:
            self.model = load_model(model_filename)

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = to_categorical(actions, num_classes=self.action_dim)
        discounted_rewards = self.compute_discounted_rewards(rewards)
        self.model.fit(states, actions, sample_weight=discounted_rewards, verbose=0)

    def act(self, state):
        policy = self.model.predict(state.reshape(1, self.state_dim))
        return np.random.choice(self.action_dim, p=policy[0])

    def compute_discounted_rewards(self, rewards):
        n = len(rewards)
        dr = np.zeros(n)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.gamma + rewards[t]
            dr[t] = running_add
        return dr

    def save_model(self, filename):
        self.model.save(filename)


def build_model(state_dim, action_dim):
    model = Sequential()
    weight_num = max(state_dim, 32)
    model.add(Dense(weight_num, input_dim=state_dim, activation="relu"))
    model.add(Dense(weight_num, activation="relu"))
    model.add(Dense(action_dim, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model
