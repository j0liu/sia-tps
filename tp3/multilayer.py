import numpy as np
import sys
from functools import partial
from plot import plotxy
import json
from utils import pad
import math
import activation_functions as af

def layer_normalize(layer_sizes : np.array):
    return list(map(lambda x : x+1, layer_sizes))

def hypercube_layers(layer_sizes : np.array):
    network_width = max(layer_sizes)
    return [network_width for _ in range(len(layer_sizes))]

class MultiLayerNetwork():
    def __init__(self, layer_sizes, activation_function, deriv_activation_function):
        self.layer_sizes = np.array(layer_normalize(layer_sizes))
        self.network_width = max(self.layer_sizes)
        self.activation_function = activation_function
        self.deriv_activation_function = deriv_activation_function

    def _initialize_weights(self, w : np.array, config : dict):
        for m in range(len(self.layer_sizes)-1): #sin contar la capa de output
            for j in range(1,self.layer_sizes[m+1]):
                w[m][j][0] = config['bias']
                if config.get('random_start', True):
                    w[m][j][1:self.layer_sizes[m]] = np.random.rand(self.layer_sizes[m]-1)


    def _backward_propagation(self, learning_rate : float, values : np.array, w : np.array, expected : np.array):
        network_width = max(self.layer_sizes)

        deltas = np.zeros((len(self.layer_sizes)-1, network_width))
        delta_ws = np.zeros((len(self.layer_sizes)-1, network_width, network_width))
        
        expected_copy = np.zeros(network_width)
        expected_copy[0] = 1
        #expected_copy2[1:] = pad(expected, network_width-1)
        expected_copy[1:1+len(expected)] = expected
        #assert np.allclose(expected_copy, expected_copy2)

        #initialize deltas
        for j in range(self.layer_sizes[-1]):
            h = np.dot(values[-2], w[-1][j])
            deltas[-1][j] = (expected_copy[j] - values[-1][j]) * self.deriv_activation_function(h)
            delta_ws[-1][j] = learning_rate * deltas[-1][j] * values[-2]

        for m in range(len(deltas)-2, -1, -1):
            for j in range(self.layer_sizes[m]):
                h = np.dot(values[m], w[m][j])

                deltas[m][j] = (np.dot(deltas[m+1], w[m+1][j])) * self.deriv_activation_function(h)
                
                delta_ws[m][j] = learning_rate * deltas[m][j] * values[m]
        return delta_ws

    def _forward_propagation(self, x : np.array, w : np.array):
        network_width = max(self.layer_sizes)
        values = np.zeros((len(self.layer_sizes), network_width))
        values[0,1:1+len(x)] = x
        values[:,0] = 1 #poner 1 a cada valor[m][0] para todo m
        for m in range(1, len(self.layer_sizes)):
            for j in range(1,self.layer_sizes[m]):
                values[m][j] = self.activation_function(np.dot(values[m-1], w[m-1][j]))
        return values
    
    def train_function(self, config : dict, inputs : np.array, expected_results : np.array, title="Sin titulo"):
        """
        inputs: np.array - matrix of shape p x n, of inputs
        layer_sizes: np.array - array of shape m, with layer sizes
        expected: np.array - array of shape p, of expected outputs
        activation_function: function - R -> R , unless normalized
        deriv_activation_function: function 
        return: np.array - tensor of shape m x wd x wd
        """
        
        p, _ = inputs.shape # p puntos en el plano, dim dimensiones

        i = 0
        w = np.zeros((len(self.layer_sizes)-1, self.network_width, self.network_width))
        self._initialize_weights(w, config)

        m = np.zeros_like(w)  # Initialize first moment vector
        v = np.zeros_like(w)  # Initialize second moment vector 

        min_error = sys.maxsize
        w_min = None
        weights_history = [w.copy()]

        while min_error > config['epsilon'] and (i < config['limit'] or config['limit'] == -1):
            mu = np.random.randint(0, p)

            values = self._forward_propagation(inputs[mu], w)

            delta_w = self._backward_propagation(config["learning_rate"], values, w, expected_results[mu])
            if config.get('optimizer', 'gds') == 'adam':
                m = config['b1'] * m + (1 - config['b1']) * (delta_w/config['learning_rate'])
                v = config['b2'] * v + (1 - config['b2']) * ((delta_w /config['learning_rate'])** 2)
                m_hat = m / (1 - config['b1'] ** (i + 1))
                v_hat = v / (1 - config['b2'] ** (i + 1))
                w += config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['e'])
            else:
                w += delta_w
            weights_history.append(w.copy())

            error = self.error_function(inputs, expected_results, w)
            if error < min_error:
                min_error = error
                print("error:", error)
                w_min = weights_history[-1]
            i += 1
        return w_min, weights_history
    
    def output_function(self, inputs : np.array, w : np.array):
        outputs = np.zeros((len(inputs), self.layer_sizes[-1]-1))
        for i, x in enumerate(inputs):
            outputs[i] = self._forward_propagation(x, w)[-1][1:self.layer_sizes[-1]]
        return outputs
        
    def denormalized_error(self, inputs : np.array, expected_results : np.array, w : np.array, denormalize_function):
        aux = self.activation_function
        self.activation_function = lambda x: denormalize_function(aux(x))
        error = self.error_function(inputs, denormalize_function(expected_results), w)
        self.activation_function = aux
        return error

    def error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        p, _ = inputs.shape # p puntos en el plano, dim dimensiones

        val = 0
        for mu in range(p):
            output = self._forward_propagation(inputs[mu], w)[-1][1:]
            for i in range(len(expected_results[mu])):
                val += 0.5 * (expected_results[mu][i] - output[i])**2
        return val
