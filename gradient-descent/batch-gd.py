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
def grad_w(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    summed = 0
    for i in range(len(x)):
        summed += (2/len(x)) * x[i] * (w*x[i] + b - y[i])

    return summed

def grad_b(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    summed = 0
    for i in range(len(y)):
        summed += (2/len(x)) * (w*x[i] + b - y[i])

    return summed

# Plot a random line
xp = np.linspace(20, 100, 1000)
w = 1.5
b = 5
yp = w*xp + b
plt.plot(xp, yp, 'g--', label="Initial guess")

# Update parameters w and b and set learning rate
alpha = 0.0002
# Iteration over epochs
for _ in range(50):
    w = w - alpha * grad_w(x, y, w, b)
    b = b - alpha * grad_b(x, y, w, b)

plt.plot(xp, w*xp + b, 'r', label="Batch fit")

# Plot the scatter dataset
plt.scatter(x, y, s=7.5, label="Data points")
plt.title("Linearly correlated dataset")
plt.legend()
plt.xlabel("$x$ axis")
plt.ylabel("$y$ axis")
plt.show()
