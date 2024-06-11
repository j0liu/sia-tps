import numpy as np
import sys
from functools import partial
import json
import math
import activation_functions as af
from network_abc import NetworkABC
from plotNetwork import plot_neural_network, create_network_gif

def layer_normalize(layer_sizes : np.array):
    return [x+1 for x in layer_sizes]

def hypercube_layers(layer_sizes : np.array):
    network_width = max(layer_sizes)
    return [network_width for _ in range(len(layer_sizes))]

class MultiLayerNetwork(NetworkABC):
    def __init__(self, layer_sizes, activation_function, deriv_activation_function, interval = None, title=""):
        super().__init__(activation_function, deriv_activation_function, interval, title)
        self.layer_sizes = np.array(layer_normalize(layer_sizes))
        self.network_width = max(self.layer_sizes)

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
        expected_copy[1:1+len(expected)] = expected

        #initialize deltas
        for j in range(self.layer_sizes[-1]):
            h = np.dot(values[-2], w[-1][j])
            deltas[-1][j] = (expected_copy[j] - values[-1][j]) * self.deriv_activation_function(h)
            delta_ws[-1][j] = learning_rate * deltas[-1][j] * values[-2]

        for m in range(len(deltas)-2, -1, -1):
            hv = np.dot(values[m], w[m].T)
            deltas[m] = (np.dot(deltas[m+1], w[m+1].T)) * self.deriv_activation_function(hv)
            delta_ws[m] = learning_rate * np.outer(deltas[m], values[m])
            # for j in range(self.layer_sizes[m]):
            #     h = np.dot(values[m], w[m][j])

            #     deltas[m][j] = (np.dot(deltas[m+1], w[m+1][j])) * self.deriv_activation_function(h)
                
            #     delta_ws[m][j] = learning_rate * deltas[m][j] * values[m]
        return delta_ws

    def _forward_propagation(self, x : np.array, w : np.array):
        network_width = max(self.layer_sizes)
        values = np.zeros((len(self.layer_sizes), network_width))
        values[0,1:1+len(x)] = x
        values[:,0] = 1 #poner 1 a cada valor[m][0] para todo m
        for m in range(1, len(self.layer_sizes)):
            values[m] = self.activation_function(np.dot(values[m-1], w[m-1].T))
            # for j in range(1,self.layer_sizes[m]):
            #     values[m][j] = self.activation_function(np.dot(values[m-1], w[m-1][j]))
        return values
    
    def train_function(self, config : dict, inputs : np.array, expected_results : np.array):
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
            # mu = np.random.randint(0, p)
            batch_mus = np.random.choice(p, min(p,config.get('batch_size', p)), replace=False)

            resultant_w = w
            for mu in batch_mus:
                values = self._forward_propagation(inputs[mu], w)
                
                delta_w = self._backward_propagation(config["learning_rate"], values, w, expected_results[mu])
                if config.get('optimizer', 'gds') == 'adam':
                    m = config['b1'] * m + (1 - config['b1']) * (delta_w/config['learning_rate'])
                    v = config['b2'] * v + (1 - config['b2']) * ((delta_w /config['learning_rate'])** 2)
                    m_hat = m / (1 - config['b1'] ** (i + 1))
                    v_hat = v / (1 - config['b2'] ** (i + 1))
                    resultant_w += config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['e'])
                else:
                    resultant_w += delta_w
                    
            w = resultant_w
            weights_history.append(w.copy())

            error = self.error_function(inputs, expected_results, w)
            if error != None and error < min_error:
                min_error = error
                print(f"{i} error:", error)
                w_min = weights_history[-1]
            i += 1
        return w_min, weights_history
    
    def output_function(self, inputs : np.array, w : np.array):
        outputs = np.zeros((len(inputs), self.layer_sizes[-1]-1))
        for i, x in enumerate(inputs):
            outputs[i] = self._forward_propagation(x, w)[-1][1:self.layer_sizes[-1]]
        return outputs


    def error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        p, _ = inputs.shape # p puntos en el plano, dim dimensiones

        sum_val = 0
        # val = 0
        # Discrete error
        for mu in range(p):
            output = self._forward_propagation(inputs[mu], w)[-1][1:]
            val = np.sum(np.abs(expected_results[mu] - np.sign(output))/2)
            sum_val += val
            if val > 1:
                return None

        # MSE
        # for mu in range(p):
        #     output = self._forward_propagation(inputs[mu], w)[-1][1:]
        #     val += 0.5 * np.sum( (expected_results[mu] - output)**2)
            # for i in range(len(expected_results[mu])):
            #     val += 0.5 * (expected_results[mu][i] - output[i])**2
        return val

    # Importante: Se asume que el autoencoder tiene una arquitecutra simetrica y de longitud impar
    def get_encoder(self, w : np.array):
        return MultiLayerNetwork([x-1 for x in self.layer_sizes[:len(self.layer_sizes)//2+1]], self.activation_function, self.deriv_activation_function, self.interval, f"{self.title} encoder"), w[:len(self.layer_sizes)//2+1]

    def get_decoder(self, w : np.array):
        return MultiLayerNetwork([x-1 for x in self.layer_sizes[len(self.layer_sizes)//2:]], self.activation_function, self.deriv_activation_function, self.interval, f"{self.title} encoder"), w[len(self.layer_sizes)//2:]

    def export_weights(self, w : np.array, filename : str):
        with open(filename, 'w+') as f:
            f.write(f"{self.layer_sizes} {self.title}, {self.interval}\n")
            for m in range(len(w)):
                for j in range(self.layer_sizes[m]):
                    f.write(" ".join(map(str, w[m][j])) + "\n")

    def import_weights(self, filename : str):
        w = np.zeros((len(self.layer_sizes)-1, self.network_width, self.network_width))
        with open(filename, 'r') as f:
            f.readline()
            for m in range(len(self.layer_sizes)-1):
                for j in range(self.layer_sizes[m]):
                    w[m][j] = list(map(float, f.readline().split()))
        return w
    
    def plot_evaluation(self, inputs, w):
        for i in inputs:
            values = self._forward_propagation(i, w)
            plot_neural_network(self.layer_sizes, w, values)
