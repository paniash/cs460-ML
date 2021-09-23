import numpy as np
import pandas as pd

# Activation function
def averaged_activation(feature, weights, bias, survival_time) -> int:
    pred = weights[0]
    for i in range(len(feature)):
        pred += (weights[i] * feature[i]) * survival_time[i]

    if pred >= 0.0:
        return 1.0
    else:
        return -1.0

#%% Averaged perceptron classifier
def averaged_correction(weights, features, labels, bias, survival, epochs=3):
    survival = np.zeros(weights.size, dtype=int)
    for _ in range(epochs):
        for i in range(len(features)):
            act = averaged_activation(features[i], weights, bias, survival)

            # Loop per feature of x
            for j in range(len(features[i])):
                if labels[i] * act < 0:
                    weights[j] = weights[j] + labels[i] * features[i][j]
                    bias = bias + labels[i]

                else:
                    survival[j] += 1

    return weights, bias, survival

#%%
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

# Initializing weights and bias (they are parameters)
weights = np.zeros(X_train[0].size, dtype=float)
bias = 0
survival = np.zeros(weights.size)

# Train the perceptron for the weights
weights, bias, survival = averaged_correction(weights, X_train, y_train, bias, survival)

# Storing predicted values
prediction = np.zeros(len(y_test))
for i in range(len(y_test)):
    prediction[i] = averaged_activation(weights, X_test[i], bias, survival)

# Calculating accuracy of vanilla perceptron
correct = 0
for i in range(len(y_test)):
    if prediction[i] == y_test[i]:
        correct += 1

print("Ratio of correct/total predictions: {}/{}".format(correct, y_test.size))
print("Accuracy = {}%".format(round(correct/y_test.size * 100, ndigits=3)))
