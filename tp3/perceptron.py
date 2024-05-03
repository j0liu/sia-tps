import numpy as np
import sys
from functools import partial
from plot import plotxy
import json
from utils import pad
import math
# from networkPlotter import plot_neural_network

with open("tp3/config.json") as f:
    config = json.load(f)


def layer_normalize(layer_sizes : np.array):
    return list(map(lambda x : x+1, layer_sizes))

def hypercube_layers(layer_sizes : np.array):
    network_width = max(layer_sizes)
    return [network_width for _ in range(len(layer_sizes))]

def simple_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones
    o = lambda x: activation_function(np.dot(x, w))
    val = 0.5 * sum((expected[mu] - o(inputs[mu]))**2 for mu in range(p))
    return val

#puntos en el plano: n = 2
#x p x n , expected p x 1
#activation_function: theta
#expected: tzeta
#se debe tomar epsilon 0 para step
def train_perceptron(config : dict, inputs : np.array, expected_results : np.array, activation_function, title="Sin titulo", error_function = simple_error, deriv_activation_function = lambda x: 1):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones

    i = 0
    w = np.concatenate((np.array([config['bias']]), (np.random.rand(dim-1)))) # pesos
    min_error = sys.maxsize
    w_hist = [w.copy()]
    w_min = w_hist[-1]
    e_list = []
    while min_error > config['epsilon'] and (i < config['limit'] or config['limit'] == -1):
        mu = np.random.randint(0, p)
        h = np.dot(inputs[mu],w)
        o = activation_function(h)

        delta_w = config['learning_rate'] * (expected_results[mu] - o) * deriv_activation_function(h) * inputs[mu]
        w += delta_w
        w_hist.append(w.copy())
        error = error_function(inputs, expected_results, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = w_hist[-1]
        i += 1
        e_list.append(error.copy())
    print("iteraciones:", i)
    # plot_decision_boundary(w_list, inputs, expected_results, title)
    # plot_errors(e_list, title)
    return w_min, w_hist



def initialize_weights(layer_sizes : np.array, w : np.array, config : dict):
    for m in range(len(layer_sizes)-1): #sin contar la capa de output
        for j in range(1,layer_sizes[m+1]):
            w[m][j][0] = config['bias']
            w[m][j][1:layer_sizes[m]] = np.random.rand(layer_sizes[m]-1)

def train_multilayer_perceptron(config : dict, inputs : np.array, layer_sizes : np.array, expected_results : np.array, activation_function, deriv_activation_function = lambda x: 1, title="Sin titulo"):
    """
    inputs: np.array - matrix of shape p x n, of inputs
    layer_sizes: np.array - array of shape m, with layer sizes
    expected: np.array - array of shape p, of expected outputs
    activation_function: function - R -> R , unless normalized
    deriv_activation_function: function 
    return: np.array - tensor of shape m x wd x wd
    """
    
    p, _ = inputs.shape # p puntos en el plano, dim dimensiones
    network_width = max(layer_sizes)

    i = 0
    w = np.zeros((len(layer_sizes)-1, network_width, network_width))
    initialize_weights(layer_sizes, w, config)

    m = np.zeros_like(w)  # Initialize first moment vector
    v = np.zeros_like(w)  # Initialize second moment vector 

    min_error = sys.maxsize
    w_min = None
    weights_history = [w.copy()]

    while min_error > config['epsilon'] and (i < config['limit'] or config['limit'] == -1):
        mu = np.random.randint(0, p)

        values = forward_propagation(inputs[mu], layer_sizes, w, activation_function)

        delta_w = backward_propagation(config["learning_rate"], values, layer_sizes, w, expected_results[mu], deriv_activation_function)
        if config['optimizer'] == 'adam':
            m = config['b1'] * m + (1 - config['b1']) * (delta_w/config['learning_rate'])
            v = config['b2'] * v + (1 - config['b2']) * ((delta_w /config['learning_rate'])** 2)
            m_hat = m / (1 - config['b1'] ** (i + 1))
            v_hat = v / (1 - config['b2'] ** (i + 1))
            w += config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['e'])
        else:
            w += delta_w
        weights_history.append(w.copy())

        error = multi_error(inputs, expected_results, layer_sizes, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = weights_history[-1]
        i += 1
    return w_min, weights_history, i


def forward_propagation(x : np.array, layer_sizes : np.array, w : np.array, activation_function):
    network_width = max(layer_sizes)
    values = np.zeros((len(layer_sizes), network_width))
    values[0,1:1+len(x)] = x
    values[:,0] = 1 #poner 1 a cada valor[m][0] para todo m
    for m in range(1, len(layer_sizes)):
        for j in range(1,layer_sizes[m]):
            values[m][j] = activation_function(np.dot(values[m-1], w[m-1][j]))
    return values

def backward_propagation(learning_rate : float, values : np.array, layer_sizes : np.array, w : np.array, expected : np.array, deriv_activation_function):
    network_width = max(layer_sizes)

    deltas = np.zeros((len(layer_sizes)-1, network_width))
    delta_ws = np.zeros((len(layer_sizes)-1, network_width, network_width))
    # expected = np.pad(expected, (0, network_width - len(expected)), 'constant')

    # expected = pad(expected, network_width-1)
    expected_copy = np.zeros(network_width)
    expected_copy[0] = 1
    expected_copy[1:] = pad(expected, network_width-1)

    #base case
    #der_act_h = np.array([deriv_activation_function(np.dot(values[-2], w[-1][i])) for i in range(layer_sizes[-1])]) #deriv_activation_function(h)
    #deltas[-1] = (expected_copy - values[-1]) * pad(der_act_h, network_width)
    #delta_ws[-1] = learning_rate * deltas[-1] * values[-2]

    #initialize deltas
    for j in range(layer_sizes[-1]):
        h = np.dot(values[-2], w[-1][j])
        deltas[-1][j] = (expected_copy[j] - values[-1][j]) * deriv_activation_function(h)
        delta_ws[-1][j] = learning_rate * deltas[-1][j] * values[-2]


    for m in range(len(deltas)-2, -1, -1):
        for j in range(layer_sizes[m]):
            h = np.dot(values[m], w[m][j])

            deltas[m][j] = (np.dot(deltas[m+1], w[m+1][j])) * deriv_activation_function(h)
            
            delta_ws[m][j] = learning_rate * deltas[m][j] * values[m]
    return delta_ws


def multi_error(inputs : np.array, expected_results : np.array, layer_sizes : np.array, w : np.array, activation_function):
    p, _ = inputs.shape # p puntos en el plano, dim dimensiones

    val = 0
    for mu in range(p):
        output = forward_propagation(inputs[mu], layer_sizes, w, activation_function)[-1][1:]
        for i in range(len(expected_results[mu])):
            val += 0.5 * (expected_results[mu][i] - output[i])**2
    # val = 0.5 * sum(sum((expected_results[mu][i] - o(inputs[mu][i]))**2 for i in range(1, len(expected_results[mu]) + 1)) for mu in range(p))
    # print(val)  
    return val