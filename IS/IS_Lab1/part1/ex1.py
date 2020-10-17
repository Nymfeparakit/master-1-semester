import csv
import warm_up
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

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

    return theta


print("Warm up excercise")
ident_matr = np.identity(5)
print(ident_matr)
input("Нажмите на любую клавишу, чтобы продолжить...")

print("Отображение данных в виде точечного графика")
X = []
Y = []
with open('ex1data1.csv') as f:
    scv_reader = csv.reader(f, delimiter=',')
    for row in scv_reader:
        X.append([row[0]])
        Y.append([row[1]])
X = np.asarray(X).astype(np.float)
Y = np.asarray(Y).astype(np.float)

fig = plt.figure()
plt.scatter(X, Y)
ax = fig.axes[0]
plt.xticks(np.arange(0, 23, step=2))
#plt.show()
input("Нажмите на любую клавишу, чтобы продолжить...")

print("Выполнение градиентного спуска")
ones_column = np.ones((97, 1))
X = np.hstack((ones_column, X))
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

#ut.computeCost(X, Y, theta)
theta = gradient_descent(X, Y, theta, alpha, iterations)
print("Значение theta, полученное методом градиентного спуска:\n", theta)

plt.plot(X[:, 1], X.dot(theta))
ax.legend(["Линейная регрессия", "Оюучающие данные"])

predict1 = np.array([1, 3.5]).dot(theta)
print("Для количества изделий = 35,000, предсказываем прибыль:", predict1)

predict2 = np.array([1, 7]).dot(theta)
print("Для количества изделий = 70,000, предсказываем прибыль:", predict2)

print("Визуализация J")
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
#theta0_vals = np.linspace(-10, 1, 10)
#theta1_vals = np.linspace(-1, 1, 10)

J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.zeros((2, 1))
        t[0, 0] = theta0_vals[i]
        t[1, 0] = theta1_vals[j]
        J_vals[i, j] = computeCost(X, Y, t)

fig = plt.figure()
ax = plt.axes(projection='3d')

#ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour(theta0_vals, theta1_vals, J_vals, stride=np.logspace(-2, 3, 20))
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')

with open("surf_data.txt", "w", newline='') as f:
    writer = csv.writer(f)
    for i in range(100):
        writer.writerow([theta0_vals[i], theta1_vals[i]])

with open('surf_values.txt', 'w', newline='') as f:
    writer = csv.writer(f)
    for i in range(100):
        row = []
        for j in range(100):
            row.append(J_vals[i, j])
        writer.writerow(row)

with open('theta_vals.txt', 'w', newline='') as f:
    writer = csv.writer(f)
    theta0 = theta.item(0)
    theta1 = theta.item(1)
    writer.writerow([theta0, theta1])

plt.show()




