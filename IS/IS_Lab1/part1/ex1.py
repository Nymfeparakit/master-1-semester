import csv
import warm_up
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from mpl_toolkits import mplot3d

print("Warm up excercise")
warm_up.execute()
input("Нажмите на любую клавишу, чтобы продолжить...")

X = []
Y = []
with open('ex1data1.csv') as f:
    scv_reader = csv.reader(f, delimiter=',')
    for row in scv_reader:
        X.append([row[0]])
        Y.append([row[1]])
X = np.asarray(X).astype(np.float)
Y = np.asarray(Y).astype(np.float)

plt.scatter(X, Y)
plt.xticks(np.arange(0, 23, step=2))
#plt.show()
input("Нажмите на любую клавишу, чтобы продолжить...")

ones_column = np.ones((97, 1))
X = np.hstack((ones_column, X))
theta = np.zeros((2, 1))
print("initial theta:", theta)

iterations = 1500
alpha = 0.01

#ut.computeCost(X, Y, theta)
ut.gradient_descent(X, Y, theta, alpha, iterations)

plt.plot(X[:, 1], X.dot(theta))
#plt.show()

predict1 = np.array([1, 3.5]).dot(theta)
print("predict1:", predict1)

predict2 = np.array([1, 7]).dot(theta)
print("predict2:", predict2)

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
        J_vals[i, j] = ut.computeCost(X, Y, t)

print("x len:", theta0_vals.size)
print("y len:", theta1_vals.size)
print("j vals len:", J_vals.size)
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

#plt.show()

print(J_vals[-10, -1])
print(J_vals[10, 4])
print(J_vals[10, -1])
print(J_vals[10, 4])




