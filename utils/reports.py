from datetime import datetime


def save_report(weights, biases, train_accuracy, test_accuracy, time_elapsed, train_size, test_size,
                generations) -> None:
    """
    Saves a report to the reports directory.
    """
    with open(f'reports/{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}.txt', 'w') as f:
        report = f'****************** The Deep Network Report ******************\n' + \
                 f'\n' + \
                 f'Weights:\n'

        for ind, weight in enumerate(weights):
            report += "W[{}] = {}\n".format(ind, weight)

        report += f'\n' + \
                  f'Bias:\n'

        for ind, bias in enumerate(biases):
            report += "b[{}] = {}\n".format(ind, bias)

        report += f'\n' + \
                  f'Train accuracy = {train_accuracy}%\n' + \
                  f'Test accuracy = {test_accuracy}%\n' + \
                  f'\n' + \
                  "Time elapsed = {:.4f}ms\n".format(time_elapsed) + \
                  f'\n' + \
                  f'Train data size = {train_size} entities\n' + \
                  f'Test data size = {test_size} entities\n' + \
                  f'Generations (epochs) of the training = {generations}\n'
        f.write(report)
