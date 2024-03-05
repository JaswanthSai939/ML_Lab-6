import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
file_path = "C:\\Users\\sai jaswanth\\Desktop\\ML_Project\\Parkinsson disease (1).csv"

if os.path.exists(file_path):
    data = pd.read_csv(file_path)  # Add this line to load the dataset

    # Perceptron class with sigmoid activation
    class Perceptron:
        def __init__(self, input_size, learning_rate, weights):
            self.input_size = input_size
            self.learning_rate = learning_rate
            self.weights = np.array(weights)
            self.errors = []

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def predict(self, inputs):
            activation = np.dot(self.weights, inputs)
            return self.sigmoid(activation)

        def train(self, inputs, target):
            prediction = self.predict(inputs)
            error = target - prediction
            self.errors.append(error)
            self.weights += self.learning_rate * error * prediction * (1 - prediction) * inputs

    # Extract inputs and targets from the dataset
    inputs = data.iloc[:, :-1].values
    targets = data.iloc[:, -1].values

    # Initial weights
    initial_weights = [0.5] * (inputs.shape[1] + 1)  # +1 for the bias term

    # Create Perceptron instance
    perceptron = Perceptron(input_size=inputs.shape[1] + 1, learning_rate=0.05, weights=initial_weights)

    # Training the perceptron
    convergence_error = 0.002
    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        total_error = 0
        for i in range(len(inputs)):
            inputs_with_bias = np.insert(inputs[i], 0, 1)  # Adding bias term
            perceptron.train(inputs_with_bias, targets[i])
            total_error += abs(perceptron.errors[-1])

        avg_error = total_error / len(inputs)

        if avg_error <= convergence_error:
            break

        iteration += 1

    # Plotting the error over epochs
    plt.plot(range(1, iteration + 1), perceptron.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Perceptron Training for Status Classification')
    plt.show()

    print(f"Number of epochs needed for convergence: {iteration}")
else:
    print(f"File not found: {file_path}")
