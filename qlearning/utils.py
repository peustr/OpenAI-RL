import numpy as np


def normalize_state(state):
    return (np.array(state) / 255).reshape(1, len(state))
