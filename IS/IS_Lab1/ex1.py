import csv
import warm_up
import matplotlib.pyplot as plt
import numpy as np
import utils as ut

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

iterations = 4
alpha = 0.01

#ut.computeCost(X, Y, theta)
ut.gradient_descent(X, Y, theta, alpha, iterations)


