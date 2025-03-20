# Pokhrel, Apar
# 1001_646_558
# 2024_09_23
# Assignment_01_01


import numpy as np
import copy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_weights(layers, x_train, seed):
    """
    Weights are initialized different;y for first layer vs the others

    """

    input_dims = x_train.shape[0]
    weights = []

    for i in range(len(layers)):
        # Number of nodes in the previous layer
        input_size = input_dims if i == 0 else layers[i - 1]

        output_size = layers[i]  # Number of nodes in the current layer
        np.random.seed(seed)
        W = np.random.randn(output_size, input_size + 1)  # adding +1 for the bias

        weights.append(W)

    return weights


def calculate_output(weights, X):

    X = np.vstack((np.ones((1, X.shape[1])), X))

    for i in range(len(weights)):
        result = np.dot(weights[i], X)  # dot product with weight matrix
        X = sigmoid(result)

        if i < len(weights) - 1:  # keep stacking ones unitl the last layer
            X = np.vstack((np.ones((1, X.shape[1])), X))
    return X


# output of a small sample with the weights
def calculate_single_output(weights, x):

    ones = np.ones((1, x.shape[0]))

    x = np.vstack((ones, x))

    for i in range(len(weights)):
        result = np.dot(weights[i], x)
        x = sigmoid(result)

        if i < len(weights) - 1:
            x = np.vstack((ones, x))
    return x


def calculate_mse_error(target, predicted):
    return np.square(np.subtract(target, predicted)).mean()


# https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__deepcopy__.html
def calculate_gradients(X, target, weights, h):
    """ """

    gradients = []

    original_weights = copy.deepcopy(weights)

    # Go through Weights, and for layer
    for l in range(len(weights)):
        # something to fill in the gradients, update the weight only after all gradients are calculated
        layer_gds = np.zeros_like(weights[l])

        # go through each of the weigt array
        for i in range(weights[l].shape[0]):
            for j in range(weights[l].shape[1]):

                initial_weight = original_weights[l][i][j]  # save the initial weight

                weights_plus = copy.deepcopy(original_weights)  # copies for +h

                weights_minus = copy.deepcopy(original_weights)  # copies for - h

                # f(w+h)
                weights_plus[l][i][j] = initial_weight + h  # Add liitle h

                output_h_plus = calculate_output(weights_plus, X)  # x_train
                mse_plus = calculate_mse_error(target, output_h_plus)  # y_train

                # f(w-h)
                weights_minus[l][i][j] = initial_weight - h  # Subtract liitle h

                output_h_minus = calculate_output(weights_minus, X)  # x_train

                mse_minus = calculate_mse_error(target, output_h_minus)  # y_train

                # not needed since updating the copies
                # weights[l][i][j] = initial_weight  # restore the weights

                # centered  difference approximation
                gradient = (mse_plus - mse_minus) / (2 * h)

                layer_gds[i][j] = gradient
        # add each of the layer gradients to the big gradient list
        gradients.append(layer_gds)

    return gradients


def multi_layer_nn(X_train, y_train, X_test, y_test, layers, alpha, epochs, h=0.00001, seed=2):
    """
    To do:  Go over each of the sample. Cannot pass 3 test cases.

    weights = initialize_weights(layers, X_train, seed)

    if epochs == 0:
        return [weights, np.array([]), calculate_output(weights, X_test)]

    mse_errors = []  # mse errors for each epoch

    for epoch in range(epochs):
        print(f"{epoch} input to calc gradient weights", weights)

        for idx in range(X_train.shape[1]):
            x_element = X_train[:,idx]]
            y_element = y_train[:,idx]]
            calculate_gra = calculate_gradients(x_element, y_element, weights, h)
            print("my gradients are:\n", calculate_gra)

            for l in range(len(weights)):
                weights[l] -= alpha * calculate_gra[l]

        print("Updated Weights:", weights)

        test_output = calculate_output(weights, X_test)

        mse_test = calculate_mse_error(test_output, y_test)

        mse_errors.append(mse_test)

        print(test_output)

    """
    weights = initialize_weights(layers, X_train, seed)

    if epochs == 0:
        return [weights, np.array([]), calculate_output(weights, X_test)]

    mse_errors = []  # mse errors for each epoch

    for epoch in range(epochs):
        # predicted_output = calculate_output(weights, X_train)
        for idx in range(X_train.shape[1]):  # Iterate over each training sample
            x_element = X_train[:, idx].reshape(-1, 1)  # Convert to column vector
            y_element = y_train[:, idx].reshape(-1, 1)

            gradients = calculate_gradients(x_element, y_element, weights, h)

            for l in range(len(weights)):
                weights[l] -= alpha * gradients[l]

        test_output = calculate_output(weights, X_test)

        mse_test = calculate_mse_error(test_output, y_test)
        mse_errors.append(mse_test)

    return [weights, np.array(mse_errors), test_output]
