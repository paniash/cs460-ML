import numpy as np
import matplotlib.pyplot as plt

## Question a
mean = np.zeros(2)
covariance = np.identity(2)

x, y = np.random.multivariate_normal(mean, covariance, 100).T
plt.scatter(x, y, s=8)
plt.title("2-dimensional Gaussian distribution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('a.png')

## Question b
# The central mean point shifts to (-1,1) from origin
mean = np.array([-1,1])
covariance = np.identity(2)

x, y = np.random.multivariate_normal(mean, covariance, 100).T
plt.scatter(x, y, s=8)
plt.title("2-dimensional Gaussian distribution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('b.png')


## Question c
mean = np.zeros(2)
covariance = 2*np.identity(2)

x, y = np.random.multivariate_normal(mean, covariance, 100).T
plt.scatter(x, y, s=8)
plt.title("2-dimensional Gaussian distribution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('c.png')


## Question d
mean = np.zeros(2)
covariance = np.array([[1,0.5], [0.5,1]])

x, y = np.random.multivariate_normal(mean, covariance, 100).T
plt.scatter(x, y, s=8)
plt.title("2-dimensional Gaussian distribution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('d.png')


## Question e
mean = np.zeros(2)
covariance = np.array([[1,-0.5], [-0.5,1]])

x, y = np.random.multivariate_normal(mean, covariance, 100).T
plt.scatter(x, y, s=8)
plt.title("2-dimensional Gaussian distribution")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.savefig('e.png')
