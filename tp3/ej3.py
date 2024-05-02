import numpy as np 
import activation_functions as af
from perceptron import train_multilayer_perceptron, multi_error, forward_propagation, layer_normalize, hypercube_layers
import json
from networkPlotter import plot_neural_network
from plot import plot_function
import math

with open("tp3/config.json") as f:
    config = json.load(f)


def output(x, layer_sizes, w, activation_function):
    return forward_propagation(x, layer_sizes, w, activation_function)[-1][1:layer_sizes[-1]]


def ejercicio_3():
    inputs = np.array([[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    expected_or = np.array([[-1], [1], [1], [-1]])
    activation_function = af.gen_tanh(config['beta'])
    activation_function_derivative = af.gen_tanh_derivative(config['beta'])

    # data = parse_to_matrices('tp3/TP3-ej3-digitos.txt')
    layer_sizes = np.array(layer_normalize([2,2,2,1]))
    w_or, weights_history = train_multilayer_perceptron(config, inputs, layer_sizes, expected_or, activation_function, activation_function_derivative)
    print("iterations:", len(weights_history))
    print("initial\n",weights_history[0])
    print("val:", multi_error(inputs, expected_or, layer_sizes, weights_history[0], activation_function))
    print("minimal\n",w_or)
    print("val:", multi_error(inputs, expected_or, layer_sizes, w_or, activation_function))
    for x in inputs:
        print("x", x[1:], "f(x)=", output(x, layer_sizes, w_or, activation_function))
    plot_neural_network(w_or, layer_sizes)
    plot_neural_network(w_or, hypercube_layers(layer_sizes))
    plot_function(activation_function, -1, 1, "tanh(x)")
    plot_function(activation_function_derivative, -1, 1, "dtanh(x)")
    print()
    

def parse_to_matrices(filepath, columns=5, rows=7):
    # Initialize a list to store the grid
    grid = []

    # Read from file
    with open(filepath, 'r') as file:
        for line in file:
            # Convert each line to a list of integers
            grid.append(list(map(int, line.strip().split())))

    # Initialize matrices list
    matrices = []
    # Loop to extract 5x7 matrices
    start_row = 0
    while start_row + rows <= len(grid):
        # Extract a 5x7 matrix
        matrix = [grid[i][0:columns] for i in range(start_row, start_row + rows)]
        matrices.append(matrix)
        start_row += rows  # Move to the next set of rows
    
    return matrices

if __name__ == "__main__":
    ejercicio_3()