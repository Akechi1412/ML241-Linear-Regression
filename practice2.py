import numpy as np
import csv

# Feature Scaling
def feature_scaling(X):
    mean = np.mean(X, axis=0)
    max = np.max(X, axis=0)
    return (X - mean) / max

# Cost Function
def cost(X, y, theta):
    m = X.shape[0]
    return 1 / (2*m) * np.sum((predict(X, theta) - y) ** 2)

# Calculate y_hat
def predict(X, theta):
    return X@theta

# Gradient Descent
def gradient_descent(X, y, theta):
    m = X.shape[0]
    theta -=  learning_rate / m * (X.T) @ ((predict(X, theta) - y))
    return theta

if __name__ == '__main__':
    #Read data from csv file
    with open('Practice2_Chapter2.csv', newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        data = np.array([row for row in csv_reader], dtype='float64')

    # Initial Values
    X = data[:, :3]
    X = feature_scaling(X)
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    y = data[:, -1].reshape(-1, 1)
    theta = np.zeros(X.shape[1], dtype='float64').reshape(-1, 1)

    learning_rate = 0.5
    epochs = 1000
    epsilon = 1e-8
    delta_j = np.Infinity

    # Training loop
    for i in range(epochs):
        if abs(delta_j) > epsilon:
            j = cost(X, y, theta)
            theta = gradient_descent(X, y, theta)
            j_new = cost(X, y, theta)
            delta_j = j_new - j
            j = j_new
            print(f'iterators: {i+1}, cost: {j}')
        else:
            break

    print(f'\ntheta: {theta.reshape(-1)}')
