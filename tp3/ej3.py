import numpy as np 
import activation_functions as af
from perceptron import train_multilayer_perceptron, multi_error, forward_propagation, layer_normalize
import json

with open("tp3/config.json") as f:
    config = json.load(f)


def output(x, layer_sizes, w, activation_function):
    return forward_propagation(x, layer_sizes, w, activation_function)[-1][1:]


def ejercicio_3():
    inputs = np.array([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    expected_or = np.array([[1], [1], [-1], [-1]])

    step_function = lambda x: 1 if x >= 0 else -1

    # data = parse_to_matrices('tp3/TP3-ej3-digitos.txt')
    layer_sizes = np.array(layer_normalize([2,2,1]))
    w_or = train_multilayer_perceptron(config, inputs, layer_sizes, expected_or, af.step)
    print(w_or)
    print("val:", multi_error(np.array([[1, -1, 1]]), np.array([[1]]), layer_sizes, w_or, af.step))
    for x in inputs:
        print("x", x, "f(x)=", output(x, layer_sizes, w_or, af.step))


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