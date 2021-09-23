import numpy as np
import matplotlib.pyplot as plt
from averaged_perceptron import *
from vanilla_perceptron import *

## Plotting accuracy as a function of number of epochs for both the perceptrons
epochs = np.arange(1, 11, 1, dtype=int)

# Loading training and testing data
df = pd.read_csv('ionosphere.data', sep=",", header=None)

# Storing all but the last element of rows as feature vector X (351 datapoints)
X = np.array([df.iloc[:, 0:-1]])[0]

# Storing the last element of each row as the label
y = df.iloc[:, -1].values

# Convert labels 'g' and 'b' to '+1' and '-1' respectively
y[y == 'g'] = +1
y[y == 'b'] = -1

# Distribute data into training and testing (200 + 151)
X_train, y_train = X[0:200, :], y[0:200]
X_test, y_test = X[200:, :], y[200:]

van

avg_prediction = np.zeros(len(y_test))
for i in range(len(y_test)):
    avg_prediction[i] = averaged_activation(weights, X_test[i], bias, survival)

# Calculating accuracy of vanilla perceptron
correct = 0
for i in range(len(y_test)):
    if avg_prediction[i] == y_test[i]:
        correct += 1
