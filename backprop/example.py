import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mean_squared_error(y_, y):
    return np.mean(np.square(y_ - y))


num_epochs = 1000
lr = 0.05

X = np.array([0.05, 0.1])
y = np.array([0.01, 0.99])
h1 = np.array([0.15, 0.2])
h2 = np.array([0.25, 0.3])
b1 = 0.35
o1 = np.array([0.4, 0.45])
o2 = np.array([0.5, 0.55])
b2 = 0.6

for epoch in range(num_epochs):
    # Forward
    # First layer
    X_ = X
    h1_net = np.dot(X_, h1) + b1
    h1_out = sigmoid(h1_net)
    h2_net = np.dot(X_, h2) + b1
    h2_out = sigmoid(h2_net)
    # Second layer
    X_ = np.array([h1_out, h2_out])
    o1_net = np.dot(X_, o1) + b2
    o1_out = sigmoid(o1_net)
    o2_net = np.dot(X_, o2) + b2
    o2_out = sigmoid(o2_net)
    # Calculate loss
    y_ = np.array([o1_out, o2_out])
    loss = mean_squared_error(y_, y)
    # Backward
    # Second layer
    dw5 = (o1_out - y[0]) * sigmoid_d(o1_net) * h1_out
    dw6 = (o1_out - y[0]) * sigmoid_d(o1_net) * h2_out
    dw7 = (o2_out - y[1]) * sigmoid_d(o2_net) * h1_out
    dw8 = (o2_out - y[1]) * sigmoid_d(o2_net) * h2_out
    # First layer
    dw1 = (
        (o1_out - y[0]) * sigmoid_d(o1_net) * o1[0] * sigmoid_d(h1_net) * X[0] +
        (o2_out - y[1]) * sigmoid_d(o2_net) * o2[0] * sigmoid_d(h1_net) * X[0]
    )
    dw2 = (
        (o1_out - y[0]) * sigmoid_d(o1_net) * o1[0] * sigmoid_d(h1_net) * X[1] +
        (o2_out - y[1]) * sigmoid_d(o2_net) * o2[0] * sigmoid_d(h1_net) * X[1]
    )
    dw3 = (
        (o1_out - y[0]) * sigmoid_d(o1_net) * o1[0] * sigmoid_d(h2_net) * X[0] +
        (o2_out - y[1]) * sigmoid_d(o2_net) * o2[0] * sigmoid_d(h2_net) * X[0]
    )
    dw4 = (
        (o1_out - y[0]) * sigmoid_d(o1_net) * o1[0] * sigmoid_d(h2_net) * X[1] +
        (o2_out - y[1]) * sigmoid_d(o2_net) * o2[0] * sigmoid_d(h2_net) * X[1]
    )
    # Update the weights
    h1[0] -= lr * dw1
    h1[1] -= lr * dw2
    h2[0] -= lr * dw3
    h2[1] -= lr * dw4
    o1[0] -= lr * dw5
    o1[1] -= lr * dw6
    o2[0] -= lr * dw7
    o2[1] -= lr * dw8
    if (epoch + 1) % 10 == 0:
        print("Epoch: {}, Loss: {}".format(epoch + 1, loss))

# print(h1[0], h1[1], h2[0], h2[1], o1[0], o1[1], o2[0], o2[1])
