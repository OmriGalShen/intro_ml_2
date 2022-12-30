import numpy as np
from matplotlib import pyplot as plt

from softsvm import softsvm


def softsvm_runner_2a(sample_size: int, repeats: int, lambda_powers, title):
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
                        label=f"train error (size={sample_size})")
    plot_error_with_bar(errors=test_errors_total, x_axis=lambda_values, color="green", ecolor="lime",
                        label=f"test error (size={sample_size})")
    plt.legend(loc="upper left")
    plt.title(f"{title} (sample size: {sample_size})")
    plt.xlabel('λ')
    plt.xscale('log')
    plt.ylabel(f"Mean error {repeats} repeats")


def softsvm_runner_2b(sample_size: int, lambda_powers, title: str):
    lambda_values = [10 ** n for n in lambda_powers]

    train_errors, test_errors = [], []
    for lambda_val in lambda_values:
        train_error, test_error = calc_train_and_test_errors(sample_size, lambda_val)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.scatter(lambda_values, train_errors, color="red", label=f"train error (size={sample_size})", zorder=10)
    plt.scatter(lambda_values, test_errors, color="gold", label=f"test error (size={sample_size})", zorder=10)
    plt.legend(loc="upper left")
    plt.title(title)
    plt.ylabel(f"Error")


def plot_error_with_bar(errors, x_axis, color: str, ecolor: str, label: str):
    avg_errors, error_bar = calc_repeats_errors(errors)
    plt.plot(x_axis, avg_errors, color=color, marker="o", label=label, zorder=1)
    plt.errorbar(x=x_axis, y=avg_errors, yerr=error_bar, fmt="none", ecolor=ecolor, capsize=2, zorder=1)


def calc_repeats_errors(errors):
    avg_errors = np.apply_along_axis(lambda e: np.mean(e), 1, errors)
    min_errors = np.apply_along_axis(lambda e: min(e), 1, errors)
    max_errors = np.apply_along_axis(lambda e: max(e), 1, errors)
    avg_distance_min = avg_errors - min_errors
    avg_distance_max = max_errors - avg_errors
    error_bar = np.vstack((avg_distance_min, avg_distance_max))
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

    # Question 2A
    # softsvm_runner_2a(sample_size=100, repeats=10, lambda_powers=range(1, 11), title="Question 2A")
    # plt.show()

    # Question 2B
    # softsvm_runner_2a(sample_size=100, repeats=10, lambda_powers=range(1, 11), title="Question 2A")
    # softsvm_runner_2b(sample_size=1000, lambda_powers=[1, 3, 5, 8], title="Question 2B")
    # plt.show()

