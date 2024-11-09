# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import math
import numpy as np
from featureExtractors import FEATURE_NAMES

PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood',
            'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            "foodCount",
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',
            'furthestFood',
            'numberAvailableActions',
            "ratioCapsuleDistance",
            "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance"
        ]

        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))

        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        "*** YOUR CODE HERE ***"
        self.input_size = len(self.features_to_use)
        self.hidden_size = 20

        # Initialize weights and biases
        # Weights are initialized using a small random value (e.g., Xavier initialization for ReLU)
        # Biases are initialized to zeros

        # Hidden layer weights and biases
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(
            2. / self.input_size)  # ReLU initialization
        self.b1 = np.zeros((self.hidden_size, 1))

        # Output layer weights and biases
        self.W2 = np.random.randn(1, self.hidden_size) * np.sqrt(2. / self.hidden_size)  # Sigmoid initialization
        self.b2 = np.zeros((1, 1))

        self.history = ...

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        vector_to_classify = feature_vector[self.features_to_use].reshape(-1, 1)  # Shape: (input_size, 1)

        # Forward pass
        z1 = np.dot(self.W1, vector_to_classify) + self.b1  # Shape: (hidden_size, 1)
        a1 = self.activationHidden(z1)  # Shape: (hidden_size, 1)
        z2 = np.dot(self.W2, a1) + self.b2  # Shape: (1, 1)
        a2 = self.activationOutput(z2)  # Shape: (1, 1)

        # Output probability
        y = a2[0, 0]
        return y

    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        return np.maximum(0, x)

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        "*** YOUR CODE HERE ***"
        return 1 / (1 + np.exp(-x))

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here.
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        y_eval = np.array(labels).reshape(1, -1)  # Shape: (1, num_samples)

        # Forward pass
        Z1 = np.dot(self.W1, X_eval.T) + self.b1  # Shape: (hidden_size, num_samples)
        A1 = self.activationHidden(Z1)  # Shape: (hidden_size, num_samples)
        Z2 = np.dot(self.W2, A1) + self.b2  # Shape: (1, num_samples)
        A2 = self.activationOutput(Z2)  # Shape: (1, num_samples)

        # Predicted labels
        predictions = (A2 >= 0.5).astype(int)  # Shape: (1, num_samples)

        # Calculate accuracy
        correct = np.sum(predictions == y_eval)
        accuracy = (correct / y_eval.size) * 100
        return accuracy

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train = trainingData[:, self.features_to_use]
        X_validate = validationData[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        X_train = X_train.T  # Shape: (input_size, num_samples)
        X_validate = X_validate.T
        y_validate = np.array(validationLabels).reshape(1, -1)
        y_train = np.array(trainingLabels).reshape(1, -1)  # Shape: (1, num_samples)

        num_samples = X_train.shape[1]
        batch_size = 64
        for iteration in range(self.max_iterations):
            # Shuffle the training data
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[:, permutation]
            y_shuffled = y_train[:, permutation]

            # Mini-batch gradient descent
            for i in range(0, num_samples, batch_size):
                end = i + batch_size
                X_batch = X_shuffled[:, i:end]
                y_batch = y_shuffled[:, i:end]

                # Forward pass
                Z1 = np.dot(self.W1, X_batch) + self.b1  # Shape: (hidden_size, batch_size)
                A1 = self.activationHidden(Z1)  # Shape: (hidden_size, batch_size)
                Z2 = np.dot(self.W2, A1) + self.b2  # Shape: (1, batch_size)
                A2 = self.activationOutput(Z2)  # Shape: (1, batch_size)

                # Compute loss (Binary Cross-Entropy)
                epsilon = 1e-8  # To prevent log(0)
                loss = - (y_batch * np.log(A2 + epsilon) + (1 - y_batch) * np.log(1 - A2 + epsilon))
                loss = np.sum(loss) / X_batch.shape[1]

                # Backward pass
                dZ2 = A2 - y_batch  # Shape: (1, batch_size)
                dW2 = np.dot(dZ2, A1.T) / X_batch.shape[1]  # Shape: (1, hidden_size)
                db2 = np.sum(dZ2, axis=1, keepdims=True) / X_batch.shape[1]  # Shape: (1, 1)

                dA1 = np.dot(self.W2.T, dZ2)  # Shape: (hidden_size, batch_size)
                dZ1 = dA1 * (Z1 > 0)  # Shape: (hidden_size, batch_size)  # ReLU derivative

                dW1 = np.dot(dZ1, X_batch.T) / X_batch.shape[1]  # Shape: (hidden_size, input_size)
                db1 = np.sum(dZ1, axis=1, keepdims=True) / X_batch.shape[1]  # Shape: (hidden_size, 1)

                # Update weights and biases
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1

            # Evaluate training and validation accuracy
            if PRINT and (iteration + 1) % 100 == 0:
                train_acc = self.evaluate(X_train.T, y_train)
                val_acc = self.evaluate(X_validate.T, y_validate)
                print(f"Iteration {iteration + 1}/{self.max_iterations} - Loss: {loss:.4f} - Training Accuracy: {train_acc:.2f}% - Validation Accuracy: {val_acc:.2f}%")


    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        "*** YOUR CODE HERE ***"
        with open(weights_path, 'w') as f:
            # Save W1
            f.write("W1\n")
            np.savetxt(f, self.W1, delimiter=',')
            # Save b1
            f.write("b1\n")
            np.savetxt(f, self.b1, delimiter=',')
            # Save W2
            f.write("W2\n")
            np.savetxt(f, self.W2, delimiter=',')
            # Save b2
            f.write("b2\n")
            np.savetxt(f, self.b2, delimiter=',')

        if PRINT:
            print(f"Weights and biases saved to {weights_path}")

    def load_weights(self, weights_path):
        """
        Loads your weights from a .model file.
        Whatever you do here should work with the formatting of your save_weights function.
        """
        "*** YOUR CODE HERE ***"
        with open(weights_path, 'r') as f:
            lines = f.readlines()

        # Initialize containers
        W1 = []
        b1 = []
        W2 = []
        b2 = []
        current_matrix = None

        for line in lines:
            line = line.strip()
            if line == "W1":
                current_matrix = W1
                continue
            elif line == "b1":
                current_matrix = b1
                continue
            elif line == "W2":
                current_matrix = W2
                continue
            elif line == "b2":
                current_matrix = b2
                continue
            else:
                current_matrix.append([float(x) for x in line.split(',')])

        # Convert lists to NumPy arrays
        self.W1 = np.array(W1)
        self.b1 = np.array(b1)
        self.W2 = np.array(W2)
        self.b2 = np.array(b2)

        if PRINT:
            print(f"Weights and biases loaded from {weights_path}")
