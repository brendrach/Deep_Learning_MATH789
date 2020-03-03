import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen
import scipy.optimize as sc

mean_1 = [10, 5]
cov_1 = [[1, 0], [0, 1]]

x1, x2 = np.random.multivariate_normal(mean_1, cov_1, 1000).T

mean_2 = [5, 0]
cov_2 = [[1, 0], [0, 1]]

x3, x4 = np.random.multivariate_normal(mean_2, cov_2, 1000).T

x_1 = np.concatenate((x1, x3), axis=0)
x_2 = np.concatenate((x2, x4), axis=0)
X = np.c_[(x_1, x_2)]

X = np.c_[np.ones((X.shape[0], 1)), X]

y = np.zeros(2000)
y[1000:len(y)] = 1

theta = np.zeros((X.shape[1], 1))

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(
        y * np.log(probability(theta, x)) + (1 - y) * np.log(
            1 - probability(theta, x)))
    return total_cost

def gradient(theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

Nfeval = 1

def rosen(X): #Rosenbrock function
    return (1.0 - X[0])**2 + 100.0 * (X[1] - X[0]**2)**2 + \
           (1.0 - X[1])**2 + 100.0 * (X[2] - X[1]**2)**2


def callbackF(Xi):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}'.format(Nfeval, Xi[0], Xi[1], Xi[2], rosen(Xi)))
    Nfeval += 1
    
def fit(x, y, theta):
    opt_weights = sc.fmin_tnc(func=cost_function, x0=theta,
                  fprime=gradient,args=(x, y.flatten()), callback = callbackF)
    return opt_weights[0]

parameters = fit(X, y, theta)

x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]

plt.figure(figsize=(10,10))
plt.plot(x1, x2, 'r*', label = "Sample 1")
plt.plot(x3, x4, 'ko', label = "Sample 2")
plt.plot(x_values, - (5.508565 + np.dot(0.948279, x_values)) / -4.920406, label='Decision Boundary -- 1% of Iterations')
plt.plot(x_values, - (4.710543 + np.dot(0.665368, x_values)) / -4.256971, label='Decision Boundary -- 10% of Iterations')
plt.plot(x_values, - (14.953597 + np.dot(-1.371814, x_values)) / -1.969248, label='Decision Boundary -- 50% of Iterations')
plt.plot(x_values, - (17.574061 + np.dot(-1.395814, x_values)) / -2.836477, label='Decision Boundary -- 75% of Iterations')
plt.plot(x_values, y_values, label='Decision Boundary -- 100% Iterations')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(0,15)
plt.ylim(-5,10)
plt.legend(loc = 'lower right')
plt.show()