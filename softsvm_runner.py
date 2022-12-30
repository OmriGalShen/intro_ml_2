import numpy as np
from matplotlib import pyplot as plt

from softsvm import softsvm


def softsvm_runner_with_repeats(sample_size: int, repeats: int, lambda_powers, title: str):
    lambda_values = [10 ** n for n in lambda_powers]

    train_errors_total, test_errors_total = [], []
    for lambda_val in lambda_values:
        train_errors, test_errors = [], []
        for _ in range(repeats):
            train_error, test_error = calc_train_and_test_errors(sample_size, lambda_val)
            train_errors.append(train_error)
            test_errors.append(test_error)
        train_errors_total.append(train_errors)
        test_errors_total.append(test_errors)

    plot_error_with_bar(errors=train_errors_total, x_axis=lambda_values, color="blue", ecolor="cyan",
                        label="train error")
    plot_error_with_bar(errors=test_errors_total, x_axis=lambda_values, color="green", ecolor="lime",
                        label="test errors")
    plt.legend(loc="upper left")
    plt.title(f"{title} (sample size: {sample_size})")
    plt.xlabel('λ')
    plt.xscale('log')
    plt.ylabel(f"Mean error {repeats} repeats")
    plt.show()


def softsvm_runner_without_repeats(sample_size: int, lambda_powers, title: str):
    lambda_values = [10 ** n for n in lambda_powers]

    train_errors, test_errors = [], []
    for lambda_val in lambda_values:
        train_error, test_error = calc_train_and_test_errors(sample_size, lambda_val)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.scatter(lambda_values, train_errors, color="blue", label="train error")
    plt.scatter(lambda_values, test_errors, color="green", label="test error")
    plt.legend(loc="upper left")
    plt.title(f"{title} (sample size: {sample_size})")
    plt.xlabel('λ')
    plt.xscale('log')
    plt.ylabel(f"Error")
    plt.show()


def plot_error_with_bar(errors, x_axis, color: str, ecolor: str, label: str):
    avg_errors, error_bar = calc_repeats_errors(errors)
    plt.plot(x_axis, avg_errors, color=color, marker="o", label=label)
    plt.errorbar(x=x_axis, y=avg_errors, yerr=error_bar, fmt="none", ecolor=ecolor, capsize=1)


def calc_repeats_errors(errors):
    min_errors = [min(curr) for curr in errors]
    max_errors = [max(curr) for curr in errors]
    avg_errors = [sum(curr) / len(curr) for curr in errors]
    min_distance = [a - b for a, b in zip(avg_errors, min_errors)]
    max_distance = [b - a for a, b in zip(avg_errors, max_errors)]
    error_bar = np.array([min_distance, max_distance])
    return avg_errors, error_bar


def calc_train_and_test_errors(sample_size: int, lambda_param):
    train_x, train_y = get_random_train_examples(sample_size)
    w = softsvm(lambda_param, train_x, train_y)

    train_error = calc_error(train_x, train_y, w)
    test_error = calc_error(testX, testy, w)

    return train_error, test_error


def calc_error(x_examples, y_examples, w):
    predicted_y = np.apply_along_axis(lambda x: int(np.sign(x @ w)), 1, x_examples)
    return np.sum(predicted_y != y_examples) / y_examples.shape[0]


def get_random_train_examples(sample_size: int):
    indices = np.random.permutation(trainX_source.shape[0])
    train_x = trainX_source[indices[:sample_size]]
    train_y = trainY_source[indices[:sample_size]]
    return train_x, train_y


if __name__ == '__main__':
    data = np.load('EX2q2_mnist.npz')
    trainX_source = data['Xtrain']
    trainY_source = data['Ytrain']
    testX = data['Xtest']
    testy = data['Ytest']
    softsvm_runner_with_repeats(sample_size=100, repeats=10, lambda_powers=range(1, 11), title="Experiments 1")
    softsvm_runner_without_repeats(sample_size=1000, lambda_powers=[1, 3, 5, 8], title="Experiments 2")
