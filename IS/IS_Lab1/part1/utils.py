import numpy as np


def computeCost(X, y, theta):

    m = len(y)
    J = 0

    for i in range(0, m):
        J += (X[i, :]@theta - y[i, :]) ** 2
    J /= 2*m

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(1, num_iters + 1):
        tmp_sum = 0
        for j in range(0, m):
            tmp_res = X[j, :]@theta - y[j, :]
            tmp_res = tmp_res * X[j, :]
            tmp_sum += tmp_res
        tmp_sum = (alpha/m)*tmp_sum
        for k in range(theta.shape[0]):
            theta[k, 0] -= tmp_sum[k]
        J_history[i - 1, 0] = computeCost(X, y, theta)

    print(J_history)
