import numpy as np 
import activation_functions as af
import json
from plotNetwork import plot_neural_network, create_network_gif
from plot import plot_function
from kfold import k_fold_cross_validation, process_k_fold_cross_categorization_results, analyze_method_categorization
from functools import partial
from multilayer import MultiLayerNetwork, hypercube_layers

def ejercicio_3_xor():
    with open("tp3/config/ej3-xor.json") as f:
        config = json.load(f)
    inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    expected_xor = np.array([[-1], [1], [1], [-1]])
    layer_sizes = [2,2,2,1]

    network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), "xor")

    w_xor, weights_history = network.train_function(config, inputs, expected_xor)
    print("iterations:", len(weights_history))
    # print("initial\n",weights_history[0])
    # print("error:", network.error_function(inputs, expected_xor, weights_history[0]))
    print("minimal\n", w_xor)
    print("error:", network.error_function(inputs, expected_xor, w_xor))
    
    for i, o in enumerate(network.output_function(inputs, w_xor)):
        print("x", inputs[i], "f(x)=", o)
    network.plot_evaluation(inputs, w_xor)
    # create_network_gif(network, weights_history, inputs[1], "xor")
    plot_neural_network(hypercube_layers(network.layer_sizes), w_xor)
    print()


def ejercicio_3_paridad():
    with open("tp3/config/ej3-par.json") as f:
        config = json.load(f)
    inputs = parse_to_matrices('tp3/TP3-ej3-digitos.txt')
    expected = np.array([[1], [0], [1], [0], [1], [0], [1], [0], [1], [0]])

    for i in range(10):
        # add to inputs
        m = parse_to_matrices(f'tp3/numeros/{i}.txt')[0:config.get('extra_groups', 10)]
        inputs = np.concatenate((inputs, m))
        current_expected = np.tile(1 if i % 2 == 0 else 0, (len(m), 1))
        # current_expected = np.zeros(10)
        # current_expected[i] = 1
        expected = np.concatenate((expected, current_expected))
    
    print(expected)
    print(len(expected))
    

    network = MultiLayerNetwork([35,10,2,1], af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1,1), "paridad")

    analyze_method_categorization(config, np.copy(inputs), expected, network, 0, 1)


def ejercicio_3_numeros():
    with open("tp3/config/ej3-digit.json") as f:
        config = json.load(f)
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

    network = MultiLayerNetwork([35,10,10,10], af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1,1), "digits")

    results = analyze_method_categorization(config, np.copy(inputs), expected, network, 0, 1)
    # option = int(input())
    
    # for i in range(10):
    #     print_num(inputs[10*(i+1)], 5, 7)
    #     print(network.output_function([inputs[10*(i+1)]], results[option][0]))



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

def eval_weight():
    with open("tp3/config/ej3-digit.json") as f:
        config = json.load(f)
    
    inputs = parse_to_matrices('tp3/TP3-ej3-digitos.txt')

    network = MultiLayerNetwork([35,10,10,10], af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1,1), "digits")
    w = network.import_weights("tp3/weights/weights_0.txt")
    print(network.output_function([inputs[0]], w))


if __name__ == "__main__":
    # import sys
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "xor":
    #         ejercicio_3_xor()
    #     elif sys.argv[1] == "paridad":
    #         ejercicio_3_paridad()
    #     elif sys.argv[1] == "numeros":
    #         ejercicio_3_numeros()
    #     elif sys.argv[1] == "ruido":
    #         ejercicio_3_generar_ruido()
    #     elif sys.argv[1] == "eval":
    #         eval_weight()
    # else:
    ejercicio_3_xor()
        # ejercicio_3_paridad()
        # ejercicio_3_numeros()
        # eval_weight()
        # ejercicio_3_generar_ruido()
