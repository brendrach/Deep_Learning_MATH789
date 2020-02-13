import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.neighbors import KNeighborsClassifier

mean_1 = [10, 0]
cov_1 = [[1, 0], [0, 1]]

x1, x2 = np.random.multivariate_normal(mean_1, cov_1, 1000).T
plt.plot(x1, x2, 'r*', label = "Sample 1")

mean_2 = [5, 0]
cov_2 = [[1, 0], [0, 1]]

x3, x4 = np.random.multivariate_normal(mean_2, cov_2, 1000).T
plt.plot(x3, x4, 'ko', label = "Sample 2")
plt.legend()

plt.axis('equal')

X1 = [x1,x2]
X1_class = np.zeros(len(x1))
X1 = np.transpose(X1)

X2 = [x3, x4]
X2_class = np.ones(len(x3))
X2 = np.transpose(X2)

X = np.concatenate((X1, X2))
Y = np.concatenate((X1_class, X2_class))

n_neighbors=5
knn = KNeighborsClassifier(n_neighbors, metric='euclidean')
knn.fit(X,Y)

xx, yy = np.meshgrid(np.arange(0, 15, 0.1),
                     np.arange(-5, 5, 0.1))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the contour map
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("k = "+str(n_neighbors))
plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn)
plt.axis('tight')
plt.show()

