# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set number of random data points
N = 100

# Define random points along x axis
x = np.random.uniform(20, 100, N)
x = x.reshape(-1,1)     # change the array into a column matrix

# Define a constant slope m
m = 2.5

# Generate a random set of closely spaced y-intercept values
c = np.random.uniform(-10, 10, N)
c = c.reshape(-1,1)     # change the array into a column matrix
y = m*x + c

# Make a linear regression model based on the datapoints
model = LinearRegression().fit(x,y)

# Generate points to be used for plotting the linear fit
X = np.linspace(20, 100, 1000)
Y = model.coef_[0][0] * X + model.intercept_[0]

# Plot the scatter dataset along with fitted line using linear regression
plt.scatter(x, m*x + c, s=7.5, label="Data points")
plt.plot(X, Y, 'r', label="Linear fit")
plt.title("Sample dataset to demonstrate linear regression")
plt.legend()
plt.xlabel("$x$ axis")
plt.ylabel("$y$ axis")
plt.show()
