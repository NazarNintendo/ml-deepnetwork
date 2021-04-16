from utils.models import Element
import numpy as np
import random

train_test_proportion = .80


def generate_random_data(size) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly generates linearly-separable data, writes it to a file and return as numpy arrays.
    :param size: amount of elements in data
    :type size: int
    :return: the numpy arrays of x and y_hat
    """
    with open('data/data.txt', 'w') as f:
        data = [Element() for _ in range(size)]
        for element in data:
            f.write(str(element))

    f.close()

    return split_data(data)


def read_from_file(path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads from file and parses data as numpy arrays.
    :param path: the path to a file
    :type path: str
    :return: the numpy arrays of x and y_hat
    """
    data = []

    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                data.append(Element(line.split(',')[0], line.split(',')[1], line.split(',')[2]))
    except IndexError:
        print(f'Error when reading from {path} - invalid syntax in line {i + 1}')
        exit(1)

    f.close()

    return split_data(data)


def split_data(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses data into X and Y_hat and splits to train and test sets.
    :param data: data
    :type data: list[Element]
    :return: training and testing sets for X and Y_hat
    """
    random.shuffle(data)

    x = [(float(element.x), float(element.y)) for element in data]
    y_hat = [int(element.value) for element in data]

    x_train = x[:int((len(x) + 1) * train_test_proportion)]
    x_test = x[int((len(x) + 1) * train_test_proportion):]
    y_hat_train = y_hat[:int((len(y_hat) + 1) * train_test_proportion)]
    y_hat_test = y_hat[int((len(y_hat) + 1) * train_test_proportion):]
    return np.array(x_train).T, np.array(x_test).T, np.array(y_hat_train), np.array(y_hat_test)


def read_for_prediction(path) -> np.ndarray:
    """
    Reads from file and parses data as a numpy array.
    :param path: the path to a file
    :type path: str
    :return: the numpy array of x
    """
    data = []

    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                data.append(Element(line.split(',')[0], line.split(',')[1], '0'))
    except IndexError:
        print(f'Error when reading from {path} - invalid syntax in line {i + 1}')
        exit(1)

    f.close()

    x = [(float(element.x), float(element.y)) for element in data]

    return np.array(x).T
