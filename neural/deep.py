from utils import *
from plotter import plot, graph
import time
import numpy as np

np.random.seed(int(time.time()))


class DeepNetwork:
    learning_rate = 0.25

    def __init__(self, filepath=None, size=100):
        # 2 neurons in the input layer.
        # 10 neurons in the hidden layer.
        # 1 neuron in the output layer.
        self.layer_sizes = [2, 10, 1]

        self.weights = []
        self.biases = []

        for index, layer_size in enumerate(self.layer_sizes):

            # Skip initializing for the input layer.
            if index == 0:
                continue

            # Initialize weight matrices and bias vectors.
            self.weights.append(np.random.randn(layer_size, self.layer_sizes[index - 1]) * 0.01)
            self.biases.append(np.zeros((layer_size, 1)))

        if filepath:
            self.X, self.X_test, self.Y_hat, self.Y_hat_test = read_from_file(filepath)
        else:
            self.X, self.X_test, self.Y_hat, self.Y_hat_test = generate_random_data(size)

    def train(self):
        """
        Trains deep neural network. Saves the results and plots them.
        """
        generations = 20000
        losses = []
        tic = time.time()

        for gen in range(generations):
            # Forward prop, backward prop and weights correction.
            A, cache_results = self.forward_propagation(self.X)
            derivatives = self.backward_propagation(cache_results, A)
            self.correct_weights(derivatives)

            loss = self.loss(self.Y_hat, A)
            losses.append(loss)

        toc = time.time()

        # Gather results after training.
        train_accuracy = self.get_accuracy(self.Y_hat, self.predict(self.X))
        test_accuracy = self.get_accuracy(self.Y_hat_test, self.predict(self.X_test))
        time_elapsed = toc - tic
        train_size = self.X.shape[1]
        test_size = self.X_test.shape[1]

        save_report(self.weights, self.biases, train_accuracy, test_accuracy, time_elapsed, train_size, test_size,
                    generations)

        graph(losses)

        X, P = self.get_data_for_decision_boundary()

        plot(self.X, self.Y_hat, X, P, 'Train Set')
        plot(self.X_test, self.Y_hat_test, X, P, 'Test Set')

    def forward_propagation(self, X) -> (np.ndarray, list):
        """
        Does forward propagation.
        :param X: input data
        :type X: numpy array (k,m)
        :return: network's output and cached intermediate results
        """
        cache_results = []
        layers_amount = len(self.weights)

        # Iterate over first L-1 layers and do RELU forward propagation.
        for index in range(layers_amount - 1):
            Z = self.z(self.weights[index], X, self.biases[index])
            A = self.relu(Z)
            cache_results.append((X, Z))
            X = A

        # Do the last sigmoid layer separately.
        Z = self.z(self.weights[layers_amount - 1], X, self.biases[layers_amount - 1])
        A = self.sigmoid(Z)
        cache_results.append((X, Z))

        return A, cache_results

    def backward_propagation(self, cache_results, A) -> list:
        """
        Does backward propagation.
        :param cache_results: cached results for derivatives calculation
        :type cache_results: list
        :param A: network's output
        :type A: numpy array(1,m)
        :return: list of derivatives as tuples
        """
        derivatives = []
        layers_amount = len(self.weights)

        # Do backward propagation for the last sigmoid layer separately.
        d_z = A - self.Y_hat
        d_w = 1 / d_z.shape[1] * np.dot(d_z, cache_results[layers_amount - 1][0].T)
        d_b = 1 / d_z.shape[1] * np.sum(d_z, axis=1, keepdims=True)
        d_a = np.dot(self.weights[layers_amount - 1].T, d_z)
        derivatives.append((d_w, d_b))

        # Iterate over first L-1 layers and do RELU backward propagation.
        for layer in reversed(range(layers_amount - 1)):
            Z = cache_results[layer][1]
            Z[Z < 0] = 0
            Z[Z > 0] = 1
            d_z = np.multiply(d_a, Z)
            d_w = 1 / d_z.shape[1] * np.dot(d_z, cache_results[layer][0].T)
            d_b = 1 / d_z.shape[1] * np.sum(d_z, axis=1, keepdims=True)
            d_a = np.dot(self.weights[layer].T, d_z)
            derivatives.append((d_w, d_b))

        return derivatives

    def correct_weights(self, derivatives) -> None:
        """
        Adds the anti-gradient to the weights and biases.
        :param derivatives: tuples with derivatives for respective layers
        :type derivatives: list
        :return: None
        """
        for i in range(len(derivatives)):
            self.weights[i] -= self.learning_rate * derivatives[len(derivatives) - i - 1][0]
            self.biases[i] -= self.learning_rate * derivatives[len(derivatives) - i - 1][1]

    def z(self, W, X, b) -> np.ndarray:
        """
        Performs a linear transformation of the data.
        :param X: input matrix
        :param W: weights matrix
        :param b: bias vector
        :type X: np.ndarray(n,m)
        :type W: np.ndarray(k,n)
        :type b: np.ndarray(k,1)
        :return: the numpy array(k,m) of linear transforms
        """
        return np.dot(W, X) + b

    def relu(self, Z) -> np.ndarray:
        """
        Applies the RELU to the Z array.
        :param Z: linear transforms Z of the data
        :type Z: np.ndarray(k,m)
        :return: the numpy array(k,m) of RELU images
        """
        Z[Z < 0] = 0
        return Z

    def sigmoid(self, Z) -> np.ndarray:
        """
        Applies the sigmoid to the Z array.
        :param Z: linear transforms Z of the data
        :type Z: np.ndarray(k,m)
        :return: the numpy array(k,m) of sigmoid images
        """
        return 1 / (1 + np.exp(-Z))

    def loss(self, Y_hat, Y) -> float:
        """
        Calculates cross-entropy averaged across a generation.
        :param Y_hat: the real values
        :type Y_hat: np.ndarray(1,n)
        :param Y: network's guessed values
        :type Y: np.ndarray(1,n)
        :return: averaged cross-entropy
        """
        Y_hat = Y_hat.reshape(1, Y_hat.shape[0])
        return np.sum(-Y_hat * np.log(Y + 1e-5) - (1 - Y_hat) * np.log(1 - Y + 1e-5)) / Y.shape[1]

    def predict(self, X) -> np.ndarray:
        """
        Processes the data on the trained network and returns a prediction array.
        :param X: input layer data
        :type X: np.ndarray(m,n)
        :return: prediction P array for X
        """
        Y, _ = self.forward_propagation(X)
        P = np.array([1 if y > 0.5 else 0 for y in Y[0]])
        return P

    def get_accuracy(self, Y_hat, P) -> float:
        """
        Compares values of Y_hat and P and returns accuracy in %.
        :param Y_hat: the real values
        :type Y_hat: np.ndarray(1,m)
        :param P: network's predicted values
        :type P: np.ndarray(1,m)
        :return: accuracy of the P
        """
        return (1 - np.sum(np.abs(Y_hat - P)) / Y_hat.shape[0]) * 100

    def get_data_for_decision_boundary(self) -> (np.ndarray, np.ndarray):
        """
        Returns X - an array of all points in [-1;1] x [-1;1] grid square.
        Returns P - an array of predictions for the array X.
        This data is used to plot the decision boundary.
        """
        x, y, X = -1., -1., []
        for i in range(200):
            for j in range(200):
                X.append([x, y])
                x += 0.01
            x = -1.
            y += 0.01
        X = np.array(X).T
        P = self.predict(X)
        return X, P

    def predict_from_file(self, filepath):
        """
        Predicts for the data in <filepath>.
        :param filepath: path of the file
        :type filepath: string
        """
        X = read_for_prediction(filepath)
        P = self.predict(X)
        print(f'Prediction array for [{filepath}]:\n{P}')
