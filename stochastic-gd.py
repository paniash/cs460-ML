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
y = m*x + c

# Calculate gradients
def grad_w(xi: float, yi: float, w: float, b: float) -> float:
    summed = 2 * xi * (w*xi + b - yi)
    return summed

def grad_b(xi: float, yi: float, w: float, b: float) -> float:
    summed = 2 * (w*xi + b - yi)
    return summed

# Plot a random line
xp = np.linspace(20, 100, 1000)
w = 3.5
b = 5
yp = w*xp + b
plt.plot(xp, yp, 'g--', label="Initial guess")

# Update parameters w and b and set learning rate
alpha = 0.0001
for i in range(10):
    for j in range(len(x)):
        w = w - alpha * grad_w(x[j], y[j], w, b)
        b = b - alpha * grad_b(x[j], y[j], w, b)

plt.plot(xp, w*xp + b, 'r', label="Batch fit")

# Plot the scatter dataset
plt.scatter(x, y, s=7.5, label="Data points")
plt.title("Linearly correlated dataset")
plt.legend()
plt.xlabel("$x$ axis")
plt.ylabel("$y$ axis")
plt.show()
