import numpy as np


def computeCost(X, y, theta):

    m = len(y)
    J = 0

    for i in range(0, m):
        if i == 0:
            print("X[i]:", X[i, :])
            print("theta:", theta)
            print("X[i]@theta:", X[i, :]@theta)
        J += (X[i, :]@theta - y[i, :]) ** 2
    J /= 2*m

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    print("theta3:", theta)
    m = len(y)
    print("m:", m)
    J_history = np.zeros((num_iters, 1))

    for i in range(1, num_iters + 1):
        print("theta4:", theta)
        tmp_sum = 0
        for j in range(0, m):
            tmp_res = X[j, :]@theta - y[j, :]
            tmp_res = tmp_res * X[j, :]
            tmp_sum += tmp_res
        print("theta5:", theta)
        print("np.transpose((alpha/m)*tmp_sum):", np.transpose((alpha/m)*tmp_sum))
        theta = np.subtract(theta, np.transpose((alpha/m)*tmp_sum))
        print("theta6:", theta)
        #J_history[i, 0] = computeCost(X, y, theta)
        cur_cost = computeCost(X, y, theta)
        print(cur_cost)

    #print(J_history)
