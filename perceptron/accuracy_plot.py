import numpy as np
import matplotlib.pyplot as plt
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
    vanilla_weights = np.zeros(X_train[0].size, dtype=float)
    vanilla_bias = 0

    vanilla_weights, vanilla_bias = correction(vanilla_weights, X_train, y_train,
            vanilla_bias, epoch)

    # Storing predicted values
    vanilla_prediction = np.zeros(len(y_test))
    for i in range(len(y_test)):
        vanilla_prediction[i] = activation(weights, X_train[i], bias)

    # Calculating accuracy of vanilla perceptron
    vanilla_correct = 0
    for i in range(len(y_test)):
        if vanilla_prediction[i] == y_test[i]:
            vanilla_correct += 1

    vanilla_accuracy.append(vanilla_correct/y_test.size * 100)

#########################
## Averaged perceptron ##
#########################
avg_accuracy = []

for epoch in epochs:
    avg_weights = np.zeros(X_train[0].size, dtype=float)
    avg_bias = 0
    avg_survival = np.zeros(weights.size)

    avg_weights, avg_bias, avg_survival = averaged_correction(avg_weights, X_train,
            y_train, avg_bias, avg_survival, epoch)

    avg_prediction = np.zeros(len(y_test))
    for i in range(len(y_test)):
        avg_prediction[i] = averaged_activation(weights, X_test[i], bias, survival)

    # Calculating accuracy of averaged perceptron
    avg_correct = 0
    for i in range(len(y_test)):
        if avg_prediction[i] == y_test[i]:
            avg_correct += 1

    avg_accuracy.append(avg_correct/y_test.size * 100)

#########################
####### PLOTTING ########
#########################
plt.plot(epochs, vanilla_accuracy, label="Vanilla perceptron")
plt.plot(epochs, avg_accuracy, label="Averaged perceptron")
plt.legend()
