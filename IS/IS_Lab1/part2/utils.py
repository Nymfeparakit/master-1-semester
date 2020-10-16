import numpy as np


def featureNormalize(X):

    X_norm = np.zeros(X.shape)
    feature_num = X.shape[1]
    mu = np.zeros((1, feature_num))
    sigma = np.zeros((1, feature_num))

    for i in range(feature_num):
        X_column = X[:, i]
        sigma[0, i] = X_column.std()
        mu[0, i] = X_column.mean()

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_norm[i, j] = (X[i, j] - mu[0, j])/sigma[0, j]

    return X_norm, mu, sigma


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

    return theta, J_history


def normalEqn(X, y):
    theta = np.linalg.pinv(X.T@X)@X.T@y
    return theta
