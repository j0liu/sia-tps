import numpy as np
import sys
from functools import partial
from plot import plotxy

#puntos en el plano: n = 2
#inputs p x n , expected p x 1
#activation_function: theta
#expected: tzeta
#se debe tomar epsilon 0 para step
def train_perceptron(config : dict, inputs : np.array, expected : np.array, activation_function, error_function, deriv_activation_function = lambda x: 1):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones

    i = 0
    w = np.concatenate((np.array([config['bias']]), (np.random.rand(dim-1)))) # pesos
    min_error = sys.maxsize
    w_min = None
    while min_error > config['epsilon'] and i < config['limit']:
        mu = np.random.randint(0, p)
        h = np.dot(inputs[mu],w)
        o = activation_function(h)

        delta_w = config['learning_rate'] * (expected[mu] - o) * deriv_activation_function(h) * inputs[mu][1:]
        w = np.concatenate((w[:1], w[1:] + delta_w))
        # error_o = 0.5 * np.sum((expected[mu] - o[mu]) ** 2)
        error = error_function(inputs, expected, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = w
        #plotxy(inputs[:,1],expected,w,delta_w)
        i += 1
    return w_min


def train_multilayer_perceptron(config : dict, inputs : np.array, layer_sizes : np.array, expected : np.array, activation_function, error_function, deriv_activation_function = lambda x: 1):
    """
    inputs: np.array - matrix of shape p x n, of inputs
    layer_sizes: np.array - array of shape m, with layer sizes
    expected: np.array - array of shape p, of expected outputs
    activation_function: function - R -> R , unless normalized
    error_function: function - 
    deriv_activation_function: function 
    return: np.array - tensor of shape m x wd x wd
    """
    
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones
    network_width = max(layer_sizes)

    i = 0
    w = np.zeros((len(layer_sizes)-1, network_width, network_width))# np.concatenate((np.array([config['bias']]), (np.random.rand(dim-1)))) # pesos

    min_error = sys.maxsize
    w_min = None

    level = 0
    while min_error > config['epsilon'] and i < config['limit']:
        mu = np.random.randint(0, p)
        h = np.dot(inputs[mu],w)
        values = forward_propagation(config, inputs, layer_sizes, w, activation_function, deriv_activation_function)
        o = values[-1]

        delta_w = config['learning_rate'] * (expected[mu] - o) * deriv_activation_function(h) * inputs[mu][1:]
        w = np.concatenate((w[:1], w[1:] + delta_w))
        # error_o = 0.5 * np.sum((expected[mu] - o[mu]) ** 2)
        error = error_function(inputs, expected, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = w
        i += 1
    return w_min

def forward_propagation(config : dict, inputs : np.array, layer_sizes : np.array, w : np.array, activation_function, deriv_activation_function):
    network_width = max(layer_sizes)
    values = np.zeros((len(layer_sizes+1, network_width)))
    values[0] = np.pad(inputs, (0, network_width - len(inputs)), 'constant')
    for m in range(1, len(layer_sizes)):
        for j in range(layer_sizes[m]):
            values[m][j] = activation_function(np.dot(values[m-1], w[m][j]))
    return values

def backward_propagation(config : dict, inputs : np.array, layer_sizes : np.array, w : np.array, expected : np.array, values : np.array, activation_function, deriv_activation_function):
    learning_rate = config['learning_rate']
    network_width = max(layer_sizes)

    deltas = np.zeros((len(layer_sizes-1, network_width)))
    delta_ws = np.zeros((len(layer_sizes-1, network_width)))
    expected = np.pad(expected, (0, network_width - len(expected)), 'constant')

    #base case
    der_act_h = np.array([deriv_activation_function(np.dot(values[-2], w[-1][i])) for i in layer_sizes[-1]]) #deriv_activation_function(h)
    deltas[-1] = (expected - values[-1]) * der_act_h
    delta_ws[-1] = learning_rate * deltas[-1] * values[-1]

    for m in range(len(deltas)-2, -1, -1):
        for j in range(layer_sizes[m]):
            h = np.dot(values[m], w[m][j])
            deltas[m][j] = (deltas[m+1] * w[m+1][j]) * deriv_activation_function(h)
            delta_ws[m][j] = learning_rate * deltas[m][j] * values[m][j]
    
    return delta_ws
    