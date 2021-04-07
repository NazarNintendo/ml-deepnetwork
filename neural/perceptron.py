from utils import *
from plotter import plot
import time
import numpy as np

np.random.seed(int(time.time()))


class Perceptron:
    learning_rate = 1

    def __init__(self, filepath=None, size=100):

        self.W = []
        self.b = []

        # Defines the structure of the deep neural network.
        layer_sizes = [5, 3, 1]

        # Initialize weight matrices and bias vectors.
        for _, layer_size in enumerate(layer_sizes):
            weight_matrix = np.random.randn(layer_size, 2) * 0.01
            bias_vector = np.zeros((layer_size, 1))

            self.W.append(weight_matrix)
            self.b.append(bias_vector)

        if filepath:
            self.X, self.X_test, self.Y_hat, self.Y_hat_test = read_from_file(filepath)
        else:
            self.X, self.X_test, self.Y_hat, self.Y_hat_test = generate_random_data(size)

    def describe_me(self):
        print(f'''
W = {self.W}
b = {self.b}
''')

    def train(self) -> None:
        """
        Performs forward propagation and weight correction over 4500 generations.
        :return: the numpy array of weights, the bias, the list of loss function values
        """
        generations = 4500

        losses = []

        tic = time.time()

        for gen in range(0, generations):
            Z = self.z(self.X)
            Y = self.sigmoid(Z)

            loss = self.loss(self.Y_hat, Y)
            losses.append(loss)

            d_loss = self.d_loss(self.Y_hat, Y)
            d_sigmoid = self.d_sigmoid(Z)
            d_product = d_loss * d_sigmoid

            d_W = self.d_W(self.X, d_product)
            d_b = self.d_b(d_product)

            self.correct_weights(d_W, d_b)

        toc = time.time()

        train_accuracy = self.test(self.X, self.Y_hat)
        test_accuracy = self.test(self.X_test, self.Y_hat_test)
        time_elapsed = toc - tic
        train_size = self.X.shape[1]
        test_size = self.X_test.shape[1]

        save_report(self.W, self.b, train_accuracy, test_accuracy, time_elapsed, train_size, test_size, generations)

        plot(self.X, self.Y_hat, self.W, self.b, 'TRAIN')
        plot(self.X_test, self.Y_hat_test, self.W, self.b, 'TEST')

    # def forward_propagation(self, generations):

    def correct_weights(self, d_w, d_b) -> None:
        """
        Backward prop - adds the anti-gradient to the weights.
        :param d_w: a vector of derivatives dL / dW averaged across a generation
        :type d_w: np.ndarray(1, m)
        :param d_b: a scalar of derivatives dL / dB averaged across a generation
        :type d_b: np.ndarray(1, 1)
        :return: None
        """
        self.W -= self.learning_rate * d_w
        self.b -= self.learning_rate * d_b

    def z(self, X) -> np.ndarray:
        """
        Performs a linear transformation of the input data.
        :param X: input layer data
        :type X: np.ndarray(m,n)
        :return: the numpy array(1,n) of linear transforms
        """
        return np.dot(self.W, X) + self.b

    def sigmoid(self, Z) -> np.ndarray:
        """
        Applies the sigmoid to the linear transforms.
        :param Z: linear transforms of the input data
        :type Z: np.ndarray(1,n)
        :return: the numpy array(1,n) of sigmoid images
        """
        return 1 / (1 + np.exp(-Z))

    def loss(self, Y_hat, Y) -> float:
        """
        Calculates cross-entropy averaged across a generation.
        :param Y_hat: the real values
        :type Y_hat: np.ndarray(1,n)
        :param Y: perceptron's guessed values
        :type Y: np.ndarray(1,n)
        :return: averaged cross-entropy
        """
        return np.sum(-Y_hat * np.log(Y) - (1 - Y_hat) * np.log(1 - Y)) / Y.shape[0]

    def d_loss(self, Y_hat, Y) -> np.ndarray:
        """
        Calculates a vector of derivatives dL / dY.
        :param Y_hat: the real values
        :type Y_hat: np.ndarray(1,n)
        :param Y: perceptron's guessed values
        :type Y: np.ndarray(1,n)
        :return: the vector of derivatives dL / dY
        """
        return - Y_hat / Y + (1 - Y_hat) / (1 - Y)

    def d_sigmoid(self, Z) -> np.ndarray:
        """
        Calculates a vector of derivatives dY / dZ.
        :param Z: linear transforms of the input data
        :type Z: np.ndarray(1,n)
        :return: the vector of derivatives dY / dZ
        """
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def d_W(self, X, dL_dZ) -> np.ndarray:
        """
        Calculates a vector of derivatives dL / dW averaged across a generation.
        :param X: input layer data
        :type X: np.ndarray(m,n)
        :param dL_dZ: element-wise multiplication of dL / dY * dY / dZ = dL / dZ
        :type dL_dZ: np.ndarray(1,n)
        :return: a vector of averaged derivatives dL / dW
        """
        return np.dot(X, dL_dZ.T) / X.shape[1]

    def d_b(self, dL_dZ) -> np.ndarray:
        """
        Calculates a vector of derivatives dL / db averaged across a generation.
        :param dL_dZ: element-wise multiplication of dL / dY * dY / dZ = dL / dZ
        :type dL_dZ: np.ndarray(1,n)
        :return: a vector of averaged derivatives dL / db
        """
        return np.sum(dL_dZ) / dL_dZ.shape[0]

    def test(self, X, Y_hat) -> float:
        """
        Processes the data on the trained perceptron and compares to true values.
        :param X: input layer data
        :type X: np.ndarray(m,n)
        :param Y_hat: the real values
        :type Y_hat: np.ndarray(1,n)
        :return: accuracy on the set
        """

        Y = self.sigmoid(self.z(X))
        P = np.array([1 if y > 0.5 else 0 for y in Y])

        rate = (1 - np.sum(np.abs(Y_hat - P)) / Y_hat.shape[0]) * 100

        return rate

    def predict(self, filepath='predict.txt') -> None:
        """
        Parses a file for supplied data and predicts according to perceptron's training.
        :param filepath:
        :type filepath: the path to a file
        :return: None
        """

        X = read_for_prediction(filepath)
        Y = self.sigmoid(self.z(X))
        P = np.array([1 if y > 0.5 else 0 for y in Y])

        print(f'Supplied data = \n{X.T}\n')
        print(f'Prediction = \n{P.reshape((P.shape[0], 1))}\n')
