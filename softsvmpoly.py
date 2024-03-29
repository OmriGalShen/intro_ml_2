import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    m, d = trainX.shape

    u = matrix(np.concatenate((np.zeros(m), (1 / m) * np.ones(m))))

    v = matrix(np.concatenate((np.zeros(m), np.ones(m))))

    G = np.array([[(1 + trainX[i] @ trainX[j]) ** k for j in range(m)] for i in range(m)])

    A = matrix(np.block(
        [[np.zeros((m, m)), np.eye(m)],
         [np.diag(trainy) @ G, np.eye(m)]]))

    H = np.block(
        [[2 * l * G, np.zeros((m, m))],
         [np.zeros((m, m)), np.zeros((m, m))]])
    H += 1e-5 * np.eye(H.shape[0])
    H = matrix(H)

    solvers.options['show_progress'] = False
    sol = solvers.qp(H, u, -A, -v)
    alpha = np.array(sol['x'][:m])
    return alpha


def simple_test():
    # load question 2 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
