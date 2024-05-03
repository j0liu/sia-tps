import random
import sys
import numpy as np
from perceptron import train_multilayer_perceptron, multi_error
def optimize_parameters(config, inputs, layer_sizes, expected_results, activation_function, deriv_activation_function):
    min_error = sys.maxsize
    min_i = sys.maxsize

    while True:
        # Initialize parameters with initial values
        config['b1'] = 0.9
        config['b2'] = 0.999
        config['e'] = 1e-8
        config['learning_rate'] = 0.08
        
        # Randomly adjust parameters within a certain range
        config['b1'] += random.uniform(-0.05, 0.05)
        config['b2'] += random.uniform(-0.05, 0.05)
        config['learning_rate'] += random.uniform(-0.05, 0.05)
        # Train the model with the updated parameters
        w_min, weights_history, i = train_multilayer_perceptron(config, inputs, layer_sizes, expected_results, activation_function, deriv_activation_function)
        
        # Check if the error is close to 0
        error = multi_error(inputs, expected_results, layer_sizes, w_min, activation_function)
        if i < config['limit'] and error < config['epsilon'] and error < min_error:
            min_i = i
            min_error = error
            print("NEW BEST PARAMETERS")
            print("NEW ERROR:", error)
            print("b1:", config['b1'])
            print("b2:", config['b2'])
            print("learning_rate:", config['learning_rate'])
            print("iterations:", i)