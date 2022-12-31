from itertools import product

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


def _soft_svm_poly(params):
    l, k = params
    alphas = softsvmpoly(l, k, trainX, trainY)

    def ret_func(x):
        a = sum([alphas[i] * (1 + trainX[i] @ x) ** k for i in range(trainX.shape[0])])
        return np.sign(a)

    return ret_func


def test_softsvmpoly(pred, testX, testY):
    count = 0
    for i in range(testX.shape[0]):
        if pred(testX[i]) != np.sign(testY[i]):
            count += 1
    return count / testY.shape[0]


def question4b_softsvmpoly_cross_validation(folds: int, lambda_values, k_values):
    lambda_k_comb = list(product(lambda_values, k_values))
    sample_fold_size = trainX.shape[0] // folds

    errors = []
    for param in lambda_k_comb:
        errs = []
        for fold_index in range(5):
            start = fold_index * sample_fold_size
            end = start + sample_fold_size
            X_validation = trainX[start:end]
            y_validation = trainY[start:end]
            X_train = np.concatenate([trainX[:start], trainX[end:]])
            y_train = np.concatenate([trainY[:start], trainY[end:]])

            predictor = _soft_svm_poly(param)
            errs.append(test_softsvmpoly(predictor, X_validation, y_validation))
        param_error = np.mean(np.array(errs))
        errors.append(param_error)
        print(f"average validation error for λ = {param[0]} k = {param[1]} is {param_error}")
    
    optimal_parameter = lambda_k_comb[np.argmin(errors)]
    print(f"The optimal parameter is {optimal_parameter}")
    predictor = _soft_svm_poly(lambda_k_comb[np.argmin(errors)])
    print(f"test error for selected parameters: {test_softsvmpoly(predictor, testX, testy)}")


def question4b_softsvm_cross_validation(lambda_values):
    fold_size = trainX.shape[0] // 5
    lambda_errors = []

    for lambda_param in lambda_values:
        errs = []
        for fold_index in range(5):
            start = fold_index * fold_size
            end = start + fold_size
            X_validation = trainX[start:end]
            y_validation = trainY[start:end]
            X_train = np.concatenate([trainX[:start], trainX[end:]])
            y_train = np.concatenate([trainY[:start], trainY[end:]])

            w = softsvm.softsvm(lambda_param, X_train, y_train)
            error = softsvm_runner.calc_error(w, X_validation, y_validation)
            errs.append(error)
        lambda_error = np.mean(np.array(errs))
        lambda_errors.append(lambda_error)
        print(f"The average validation error for {lambda_param} is {lambda_error}")
    print(f"The selected prameter is {lambda_values[np.argmin(lambda_errors)]}")
    w = softsvm.softsvm(lambda_values[np.argmin(lambda_errors)], trainX, trainY)
    test_error = softsvm_runner.calc_error(testX, testy, w)
    print(f"test error for selected parameters: {test_error}")


if __name__ == '__main__':
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testy = data['Ytest']

    # question_4a()

    question_4b()
