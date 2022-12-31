import numpy as np
from matplotlib import pyplot as plt

import softsvm
import softsvm_runner
from softsvmpoly import softsvmpoly


def question_4a():
    x = trainX[:, 0]
    y = trainX[:, 1]
    plt.scatter(x, y, c=trainY, cmap='plasma')
    plt.title('Question 4A - training set plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def question_4b():
    lambda_values = [1, 10, 100]
    k_values = [2, 5, 8]
    folds = 5
    question4b_softsvmpoly_cross_validation(folds, lambda_values, k_values)
    question4b_softsvm_cross_validation(folds, lambda_values)


def question4b_softsvmpoly_cross_validation(folds: int, lambda_values, k_values):
    print(f"------------------------")
    print(f"Running cross validation {folds} folds on softsvmpoly")
    print(f"------------------------")
    optimal_k, optimal_lambda = None, None
    min_error = 1.1
    m = trainX.shape[0]
    for k in k_values:
        for lambda_param in lambda_values:
            fold_errors = []
            for fold_indices in np.array_split(np.arange(m), folds):
                fold_x = trainX[fold_indices]
                fold_y = trainY[fold_indices]
                fold_error = run_softsvmpoly(lambda_param, k, fold_x, fold_y)
                fold_errors.append(fold_error)

            fold_mean_error = np.mean(fold_errors)
            print(f"k = {k}, lambda = {lambda_param}, average validation error = {fold_mean_error}")
            if fold_mean_error < min_error:
                min_error = fold_mean_error
                optimal_k, optimal_lambda = k, lambda_param

    print(f"------- Results -----------")
    print(f"Optimal parameters: k = {optimal_k}, lambda = {optimal_lambda} ")
    test_error = run_softsvmpoly(optimal_lambda, optimal_k, trainX, trainY)
    print(f"k = {optimal_k}, lambda = {optimal_lambda} test error: {test_error}")
    print(f"---------------------------")
    print()


def run_softsvmpoly(lambda_param, k, examples_x, examples_y):
    alphas = softsvmpoly(lambda_param, k, examples_x, examples_y)
    predicted_y = np.apply_along_axis(lambda x_point: calc_predicted_y(alphas, x_point, k, examples_x), 1, examples_x)
    predicted_y = predicted_y.reshape((examples_y.shape[0]))
    return np.sum(predicted_y != examples_y) / examples_y.shape[0]


def calc_predicted_y(alphas, x_point, k, x_examples):
    return np.sign(sum([alphas[i] * (1 + x_examples[i] @ x_point) ** k for i in range(x_examples.shape[0])]))


def question4b_softsvm_cross_validation(folds: int, lambda_values):
    print(f"------------------------")
    print(f"Running cross validation {folds} folds on softsvm:")
    print(f"------------------------")
    optimal_lambda = None, None
    min_error = 1.1
    for lambda_param in lambda_values:
        fold_errors = []
        for fold_indices in np.array_split(np.arange(trainX.shape[0]), folds):
            fold_x = trainX[fold_indices]
            fold_y = trainY[fold_indices]
            w = softsvm.softsvm(lambda_param, fold_x, fold_y)
            fold_error = softsvm_runner.calc_error(fold_x, fold_y, w)
            fold_errors.append(fold_error)

        fold_mean_error = np.mean(fold_errors)
        print(f"lambda = {lambda_param}, average validation error = {fold_mean_error}")
        if fold_mean_error < min_error:
            min_error = fold_mean_error
            optimal_lambda = lambda_param
    print(f"------- Results -----------")
    print(f"Optimal lambda = {optimal_lambda}")
    w = softsvm.softsvm(optimal_lambda, testX, testy)
    test_error = softsvm_runner.calc_error(testX, testy, w)
    print(f"lambda = {optimal_lambda} test error: {test_error}")
    print(f"--------------------------")
    print()


def question_4e():
    lambda_param = 100
    k_values = [3, 5, 8]

    for k in k_values:
        alphas = softsvmpoly(lambda_param, k, trainX, trainY)
        grid_bounds = np.linspace(-1, 1, 101)
        grid = [[color_func(alphas, k, i, j) for j in grid_bounds] for i in grid_bounds]

        plt.imshow(grid,
                   extent=[-1, 1, -1, 1],
                   cmap=plt.cm.RdBu)
        plt.title(f"Question 4e (lambda = {lambda_param}, k = {k})")
        plt.show()


def color_func(alphas, k, i, j):
    red, blue = (255, 0, 0), (0, 0, 255)
    if int(calc_predicted_y(alphas, (i, j), k, trainX)) == 1:
        return blue
    else:
        return red


def question_4f():
    pass


if __name__ == '__main__':
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testy = data['Ytest']

    # question_4a()
    # question_4b()
    # question_4e()
    # question_4f()
