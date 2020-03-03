import numpy as np
import matplotlib.pyplot as plt


mean_1 = [8, 6]
cov_1 = [[2, 0], [0, 2]]

x1, x2 = np.random.multivariate_normal(mean_1, cov_1, 500).T
plt.plot(x1, x2, 'r*', label = "Sample 1")

mean_2 = [4, 2]
cov_2 = [[2, 0], [0, 2]]

x3, x4 = np.random.multivariate_normal(mean_2, cov_2, 500).T
plt.plot(x3, x4, 'ko', label = "Sample 2")

plt.axis('equal')

x_1 = np.concatenate((x1, x3), axis=0)
x_2 = np.concatenate((x2, x4), axis=0)
X = np.c_[(x_1, x_2)]

#print(X)

y = np.zeros(1000)
y[500:len(y)] = 0


def logistic_regression(X, y, alpha):
    n = X.shape[1]
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)
    theta = np.zeros(n+1)
    h = hypothesis(theta, X, n)
    theta, theta_history, cost = Gradient_Descent(theta, alpha, 10000, h, X, y, n)
    return theta, theta_history, cost

def Gradient_Descent(theta, alpha, num_iters, h, X, y, n):
    theta_history = np.ones((num_iters,n+1))
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n+1):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h - y) * X.transpose()[j])
        theta_history[i] = theta
        h = hypothesis(theta, X, n)
        cost[i] = (-1/X.shape[0]) * sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    theta = theta.reshape(1,n+1)
    return theta, theta_history, cost

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n+1)
    for i in range(0,X.shape[0]):
        h[i] = 1 / (1 + np.exp(-float(np.matmul(theta, X[i]))))
    h = h.reshape(X.shape[0])
    return h


theta, theta_history, cost = logistic_regression(X, y, 0.01)

print(theta)

theta = theta.reshape(3)

x_values = [np.min(X[:, 0] - 2), np.max(X[:, 1] + 2)]
y_values =  -(theta[0] + np.dot(theta[1],x_values)) / theta[2]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.legend()
plt.show()