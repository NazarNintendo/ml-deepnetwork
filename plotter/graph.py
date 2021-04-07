import matplotlib.pyplot as plt
import numpy as np


def plot(X, Y, W, b, title):
    scatter(X.T.tolist(), Y.tolist(), title)
    line(W, b)
    show()


def scatter(X, Y, title):
    x_orange = []
    y_orange = []
    x_blue = []
    y_blue = []
    for ind, x in enumerate(X):
        if Y[ind]:
            x_orange.append(x[0])
            y_orange.append(x[1])
        else:
            x_blue.append(x[0])
            y_blue.append(x[1])

    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    plt.scatter(x_orange, y_orange, c='#ff9933', s=1)
    plt.scatter(x_blue, y_blue, c='#00e6e6', s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)


def line(W, b):
    x = np.linspace(0, 1, 1000)
    plt.plot(x, -(W[0] * x + b) / W[1])


def show():
    plt.show()


def graph(losses):
    axes = plt.gca()
    axes.set_xlim([0, 1000])
    axes.set_ylim([0, 1])
    plt.plot(losses, c='r')
