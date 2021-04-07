from datetime import datetime


def save_report(W, b,  train_accuracy, test_accuracy, time_elapsed, train_size, test_size, generations) -> None:
    """
    Saves a report to the reports directory.
    """
    with open(f'reports/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.txt', 'w') as f:
        f.write(
            f'****************** The Perceptron Report ******************\n'
            f'\n'
            f'Weights:\n' +
            "w1 = {:.4f}\n".format(W[0] / W[1]) +
            "w2 = {:.4f}\n".format(W[1] / W[1]) +
            f'\n'
            f'Bias:\n' +
            "b = {:.4f}\n".format(b / W[1]) +
            f'\n'
            "The general form of the equation of a straight line:\n\t {:.4f}x + {:.4f}y + {:.4f} = 0\n"
            .format(W[0] / W[1], W[1] / W[1], b / W[1]) +
            f'\n'
            "The cartesian form of the equation of a straight line:\n\t y = {:.4f}x + {:.4f}\n"
            .format(-W[0] / W[1], -b / W[1]) +
            f'\n'
            f'Train accuracy = {train_accuracy}%\n'
            f'Test accuracy = {test_accuracy}%\n'
            f'\n' +
            "Time elapsed = {:.4f}ms\n".format(time_elapsed) +
            f'\n'
            f'Train data size = {train_size} entities\n'
            f'Test data size = {test_size} entities\n'
            f'Generations (epochs) of the training = {generations}\n'
        )