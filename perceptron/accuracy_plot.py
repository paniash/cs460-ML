import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from averaged_perceptron import averaged_activation, averaged_correction
from vanilla_perceptron import activation, correction

## Plotting accuracy as a function of number of epochs for both the perceptrons
epochs = np.arange(1, 101, 1, dtype=int)

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

#########################
## Vanilla perceptron ###
#########################
vanilla_accuracy = []

for epoch in epochs:
    weights = np.zeros(X[0].size, dtype=float)
    bias = 0

    # Train the perceptron for the weights
    weights, bias = correction(weights, X_train, y_train, bias, epoch)

    # Storing predicted values
    prediction = np.zeros(len(y_test))
    for i in range(len(y_test)):
        prediction[i] = activation(weights, X_test[i], bias)

    # Calculating accuracy of vanilla perceptron
    correct = 0
    for i in range(len(y_test)):
        if prediction[i] == y_test[i]:
            correct += 1

    vanilla_accuracy.append(correct/y_test.size * 100)


#########################
## Averaged perceptron ##
#########################
avg_accuracy = []

for epoch in epochs:
    weights = np.zeros(X_train[0].size, dtype=float)
    bias = 0
    survival = np.zeros(weights.size)

    # Train the perceptron for the weights
    weights, bias, survival = averaged_correction(weights, X_train, y_train, bias,
            survival, epoch)

    # Storing predicted values
    prediction = np.zeros(len(y_test))
    for i in range(len(y_test)):
        prediction[i] = averaged_activation(weights, X_test[i], bias, survival)

    # Calculating accuracy of vanilla perceptron
    correct = 0
    for i in range(len(y_test)):
        if prediction[i] == y_test[i]:
            correct += 1

    avg_accuracy.append(correct/y_test.size * 100)



#########################
####### PLOTTING ########
#########################
plt.plot(epochs, vanilla_accuracy, label="Vanilla perceptron")
plt.plot(epochs, avg_accuracy, label="Averaged perceptron")
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy of vanilla vs averaged perceptron")
plt.show()
