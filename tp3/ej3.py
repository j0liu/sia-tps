import numpy as np 
import activation_functions as af
from perceptron import train_multilayer_perceptron, forward_propagation
import json

with open("tp3/config.json") as f:
    config = json.load(f)

def multi_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones
    
    o = lambda x: activation_function(np.dot(x, w)) + config["bias"]
    val = 0.5 * sum((expected[mu] - o(inputs[mu]))**2 for mu in range(p))
    return val

def ejercicio_3():
    inputs = np.array([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    expected_or = np.array([[1], [1], [-1], [-1]])

    data = parse_to_matrices('tp3\TP3-ej3-digitos.txt')
    w_or = train_multilayer_perceptron(config, inputs, [3,2,1], expected_or, af.id)
    print(w_or)


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