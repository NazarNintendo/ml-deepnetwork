import matplotlib.pyplot as plt


def plot(X, Y, Xp, Pp, title):
    """
    Plots the data and decision boundary.
    """
    scatter(Xp.T.tolist(), Pp.T.tolist(), '#FDE8D2', '#D0E9E9', title)
    scatter(X.T.tolist(), Y.tolist(), '#ff9933', '#00e6e6', title)
    plt.show()


def scatter(X, Y, color1, color2, title):
    """
    Scatters the points from X and Y on the plot.
    """
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
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.scatter(x_orange, y_orange, c=color1, s=15)
    plt.scatter(x_blue, y_blue, c=color2, s=15)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)


def graph(losses):
    """
    Plots the graph of the loss function.
    """
    axes = plt.gca()
    axes.set_xlim([0, 20000])
    axes.set_ylim([0, 1])
    plt.plot(losses, c='r')
    plt.show()
