# Create dataset that has some linear correlation and plot the dataset.

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Set number of random data points
N = 100

# Define random points along x axis
x = np.random.uniform(20, 100, N)

# Define a constant slope m
m = 2.5

# Generate a random set of closely spaced y-intercept values
c = np.random.uniform(-10, 10, N)

# Plot the scatter dataset
plt.scatter(x, m*x + c, s=7.5, label="Data points")
plt.title("Linearly correlated dataset")
plt.legend()
plt.xlabel("$x$ axis")
plt.ylabel("$y$ axis")
plt.show()
