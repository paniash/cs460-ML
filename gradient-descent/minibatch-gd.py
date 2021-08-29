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
w = 3.5
b = 5
yp = w*xp + b
plt.plot(xp, yp, 'g--', label="Initial guess")

# Divide the dataset into 5 minibatches
x1 = x[:int(len(x)/5)]
y1 = m*x1 + c[:int(len(c)/5)]
x2 = x[int(len(x)/5):int(2*len(x)/5)]
y2 = m*x2 + c[int(len(x)/5):int(2*len(x)/5)]
x3 = x[int(2*len(x)/5):int(3*len(x)/5)]
y3 = m*x3 + c[int(2*len(x)/5):int(3*len(x)/5)]
x4 = x[int(3*len(x)/5):int(4*len(x)/5)]
y4 = m*x4 + c[int(3*len(x)/5):int(4*len(x)/5)]
x5 = x[int(4*len(x)/5):]
y5 = m*x5 + c[int(4*len(x)/5):]
p = [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]

# Update parameters w and b and set learning rate
alpha = 0.0001
# Iteration over epochs
for i in range(5):
    # Iteration over mini-batches
    for j in p:
        w = w - alpha * grad_w(j[0], j[1], w, b)
        b = b - alpha * grad_b(j[0], j[1], w, b)

plt.plot(xp, w*xp + b, 'r', label="Batch fit")

# Plot the scatter dataset
plt.scatter(x, y, s=7.5, label="Data points")
plt.title("Linearly correlated dataset")
plt.legend()
plt.xlabel("$x$ axis")
plt.ylabel("$y$ axis")
plt.show()
