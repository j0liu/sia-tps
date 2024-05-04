import numpy as np 
import activation_functions as af
from perceptron import train_multilayer_perceptron, multi_error, forward_propagation, layer_normalize, hypercube_layers
import json
from plotNetwork import plot_neural_network
from plot import plot_function
from kfold import analyze_method, k_fold_cross_validation, process_k_fold_cross_validation_results
from functools import partial

with open("tp3/config.json") as f:
    config = json.load(f)


def output(x, layer_sizes, w, activation_function):
    return forward_propagation(x, layer_sizes, w, activation_function)[-1][1:layer_sizes[-1]]


def ejercicio_3_xor():
    inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    expected_or = np.array([[-1], [1], [1], [-1]])
    activation_function = af.gen_tanh(config['beta'])
    activation_function_derivative = af.gen_tanh_derivative(config['beta'])
    layer_sizes = np.array(layer_normalize([2,2,2,1]))

    w_or, weights_history = train_multilayer_perceptron(config, inputs, layer_sizes, expected_or, activation_function, activation_function_derivative)
    print("iterations:", len(weights_history))
    print("initial\n",weights_history[0])
    print("val:", multi_error(inputs, expected_or, layer_sizes, weights_history[0], activation_function))
    print("minimal\n",w_or)
    print("val:", multi_error(inputs, expected_or, layer_sizes, w_or, activation_function))
    for x in inputs:
        print("x", x, "f(x)=", output(x, layer_sizes, w_or, activation_function))
    plot_neural_network(w_or, layer_sizes)
    plot_neural_network(w_or, hypercube_layers(layer_sizes))
    plot_function(activation_function, -1, 1, "tanh(x)")
    plot_function(activation_function_derivative, -1, 1, "dtanh(x)")
    print()
    

def ejercicio_3_paridad():
    inputs = parse_to_matrices('tp3/TP3-ej3-digitos.txt')
    expected = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])
    activation_function = af.gen_tanh(config['beta'])
    activation_function_derivative = af.gen_tanh_derivative(config['beta'])

    layer_sizes = np.array(layer_normalize([35,2,2,1]))
    train_function = partial(train_multilayer_perceptron, layer_sizes=layer_sizes)
    error_function = partial(multi_error, layer_sizes=layer_sizes)


    # analyze_method(config, np.copy(inputs), expected, activation_function, activation_function_derivative, train_function, -1, 1, error_function, "paridad")
    multi_results = k_fold_cross_validation(config, train_function, inputs, expected, activation_function, error_function, "multi", deriv_activation_function=activation_function_derivative)
    process_k_fold_cross_validation_results(multi_results, activation_function, af.id, error_function, "multi")

def ejercicio_3_numeros():
    inputs = parse_to_matrices('tp3/TP3-ej3-digitos.txt')

    expected = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    for i in range(10):
        # add to inputs
        m = parse_to_matrices(f'tp3/numeros/{i}.txt')
        inputs = np.concatenate((inputs, m))
        current_expected = np.zeros(10)
        current_expected[i] = 1
        expected = np.concatenate((expected, np.tile(current_expected, (len(m), 1))))

    activation_function = af.gen_tanh(config['beta'])
    activation_function_derivative = af.gen_tanh_derivative(config['beta'])

    layer_sizes = np.array(layer_normalize([35,10,10]))
    train_function = partial(train_multilayer_perceptron, layer_sizes=layer_sizes)
    error_function = partial(multi_error, layer_sizes=layer_sizes)


    # analyze_method(config, np.copy(inputs), expected, activation_function, activation_function_derivative, train_function, -1, 1, error_function, "paridad")
    multi_results = k_fold_cross_validation(config, train_function, inputs, expected, activation_function, error_function, "multi", deriv_activation_function=activation_function_derivative)
    process_k_fold_cross_validation_results(multi_results, activation_function, af.id, error_function, "multi")



def ejercicio_3_generar_ruido():
    inputs = parse_to_matrices('tp3/TP3-ej3-digitos.txt')

    # Generate noise
    print (inputs)
    
    noise = np.random.normal(0, 0.25, inputs.shape)
    # noisy_inputs = np.clip(inputs + noise, 0, 1)

    for i, num_pixels in enumerate(inputs):
        with open(f"tp3/numeros/{i}.txt", 'w') as file:
            for _ in range(10):
                noise = np.random.normal(0, 0.25, num_pixels.shape)
                noisy_inputs = np.clip(num_pixels + noise, 0, 1)
                for j, pixel in enumerate(noisy_inputs):
                    file.write(str(round(pixel, 3)) + " ")
                    if j % 5 == 4   :
                        file.write("\n")



def parse_to_matrices(filepath, columns=5, rows=7):
    # Initialize a list to store the grid
    grid = []

    # Read from file
    with open(filepath, 'r') as file:
        for line in file:
            # Convert each line to a list of integers
            grid.append(list(map(float, line.strip().split())))

    # Initialize matrices list
    matrices = []
    # Loop to extract 5x7 matrices
    start_row = 0
    while start_row + rows <= len(grid):
        # Extract a 5x7 matrix
        matrix = [grid[i][0:columns] for i in range(start_row, start_row + rows)]
        matrix = np.array(matrix).flatten()
        matrices.append(matrix)
        start_row += rows  # Move to the next set of rows

    return np.array(matrices)

def print_num(number_array : np.array, width, height):
    #print number array
    for i in range(height):
        for j in range(width):
            pixel = number_array[i * width + j]
            if pixel > 0.75:
                character = "▧"
            elif pixel > 0.5:
                character = "⊡" #×
            elif pixel > 0.25:
                character = "∙"
            else:
                character = " "
            print(character, end = " ")
        print()

if __name__ == "__main__":
    ejercicio_3_xor()
    # ejercicio_3_paridad()
    # ejercicio_3_numeros()
    # ejercicio_3_generar_ruido()
