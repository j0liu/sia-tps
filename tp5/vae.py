import numpy as np
import sys
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

class ErrorType(object):
    DISCRETE = 0
    MSE = 1

class VAENetwork(NetworkABC):
    def __init__(self, layer_sizes, activation_function, deriv_activation_function, error_type = ErrorType.MSE, interval = None, title=""):
        super().__init__(activation_function, deriv_activation_function, interval, title)
        error_function_map = {
            ErrorType.DISCRETE : self.discrete_error_function,
            ErrorType.MSE : self.mse_error_function
        }
        self.error_type = error_type
        self.error_function = error_function_map[error_type]
        self.layer_sizes = np.array(layer_normalize(layer_sizes))
        self.network_width = max(self.layer_sizes)
        self.stochastic_layer = len(layer_sizes)//2

    def gen_layers(input_len: int, latent_len: int, encode_layers: list[int]):
        return [input_len, *encode_layers, latent_len*2, latent_len, *(encode_layers[::-1]), input_len]
    

    def _initialize_weights(self, w : np.array, config : dict):
        for m in range(len(self.layer_sizes)-1):
            for j in range(1, self.layer_sizes[m+1]):
                w[m][j][0] = config.get('bias', 0.1)
                if config.get('random_start', True):
                    # Xavier initialization
                    limit = np.sqrt(6 / (self.layer_sizes[m] + self.layer_sizes[m+1]))
                    w[m][j][1:self.layer_sizes[m]] = np.random.uniform(-limit, limit, size=self.layer_sizes[m]-1)


    def _backward_propagation(self, learning_rate: float, values: np.array, w: np.array, expected: np.array):
        deltas = np.zeros((len(self.layer_sizes) - 1, self.network_width))
        delta_ws = np.zeros((len(self.layer_sizes) - 1, self.network_width, self.network_width))
        
        expected_copy = np.zeros(self.network_width)
        expected_copy[0] = 1
        expected_copy[1:1 + len(expected)] = expected

        # Compute deltas for the output layer
        for j in range(1, self.layer_sizes[-1]):
            h = np.dot(values[-2], w[-1][j])
            deltas[-1][j] = (expected_copy[j] - values[-1][j]) * self.deriv_activation_function(h)
            delta_ws[-1][j] = learning_rate * deltas[-1][j] * values[-2]

        # Compute deltas for the hidden layers
        for m in range(len(deltas) - 2, -1, -1):
            if m == self.stochastic_layer:
                for j in range(self.layer_sizes[self.stochastic_layer]):
                    deltas[m][j] = np.sum(deltas[m+1]) #???
            else:
                for j in range(1, self.layer_sizes[m + 1]):
                    h = np.dot(values[m], w[m][j])
                    deltas[m][j] = np.dot(deltas[m + 1][1:self.layer_sizes[m + 2]], w[m + 1][1:self.layer_sizes[m + 2], j]) * self.deriv_activation_function(h)
                    delta_ws[m][j] = learning_rate * deltas[m][j] * values[m]
        
        # Gradient clipping to avoid exploding gradients
        delta_ws = np.clip(delta_ws, self.interval[0], self.interval[1])
        
        return delta_ws



    def _forward_propagation(self, x : np.array, w : np.array):
        network_width = max(self.layer_sizes)
        values = np.zeros((len(self.layer_sizes), network_width))
        values[0,1:1+len(x)] = x
        values[:,0] = 1
        eps = np.random.standard_normal()
        for m in range(1, len(self.layer_sizes)):
            if m == self.stochastic_layer:
                for i in range(self.layer_sizes[self.stochastic_layer]):
                    values[m][i] = eps * values[m-1][i*2] + values[m-1][i*2+1] # epsilon * sigma + mu
            else:
                values[m] = self.activation_function(np.dot(values[m-1], w[m-1].T))
        return values

    def forward_propagation(self, x : np.array, w : np.array):
        vals, _ = self._forward_propagation(x, w)
        # l_pos = len(vals)//2+1
        # print("inside forward")
        # print(vals[l_pos][1:self.layer_sizes[l_pos]])
        return vals[-1][1:self.layer_sizes[-1]]

    def train_function(self, config: dict, inputs: np.array, expected_results: np.array):
        """
        inputs: np.array - matrix of shape p x n, of inputs
        layer_sizes: np.array - array of shape m, with layer sizes
        expected: np.array - array of shape p, of expected outputs
        activation_function: function - R -> R , unless normalized
        deriv_activation_function: function 
        return: np.array - tensor of shape m x wd x wd
        """
        p, _ = inputs.shape
        i = 0
        w = np.zeros((len(self.layer_sizes)-1, self.network_width, self.network_width))
        self._initialize_weights(w, config)

        m = np.zeros_like(w)
        v = np.zeros_like(w)

        min_error = sys.maxsize
        w_min = None
        weights_history = [w.copy()]
        error_history = []

        learning_rate = config['learning_rate']
        learning_rate_decay = config.get('learning_rate_decay', 0.99)
        patience_counter = 0
        try:
            while min_error > config['epsilon'] and (i < config['limit'] or config['limit'] == -1):
                batch_mus = np.random.choice(p, min(p, config.get('batch_size', p)), replace=False)
                resultant_w = w.copy()
                for mu in batch_mus:
                    values, eps = self._forward_propagation(inputs[mu], w)
                    delta_w = self._backward_propagation(learning_rate, values, w, expected_results[mu])
                    
                    if config.get('optimizer', 'adam') == 'adam':
                        m = config['b1'] * m + (1 - config['b1']) * delta_w
                        v = config['b2'] * v + (1 - config['b2']) * (delta_w ** 2)
                        m_hat = m / (1 - config['b1'] ** (i + 1))
                        v_hat = v / (1 - config['b2'] ** (i + 1))
                        delta_w = learning_rate * m_hat / (np.sqrt(v_hat) + config['e'])
                    
                    # Gradient clipping
                    delta_w = np.clip(delta_w, self.interval[0], self.interval[1])
                    resultant_w += delta_w
                w = resultant_w
                error = self.error_function(inputs, expected_results, w)
                error_history.append(error)
                if error < min_error:
                    patience_counter = 0
                    min_error = error
                    print(f"{i} error:", error)
                    w_min = w.copy()

                # Learning rate decay
                learning_rate *= learning_rate_decay

                if learning_rate < config.get('min_learning_rate', 0.0001):
                    patience_counter += 1
                    print("Learning rate reset")
                    if patience_counter >= 5:
                        patience_counter = 0
                        learning_rate = config['max_learning_rate']
                        print(f'Momentum increased to {learning_rate}')
                    else:
                        learning_rate = config['learning_rate']

                i += 1
        except KeyboardInterrupt:
            pass
        return w_min, weights_history

    
    def non_lazy_discrete_error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        p, _ = inputs.shape # p puntos en el plano, dim dimensiones
        sum_val = 0
        # Discrete error
        for mu in range(p):
            output = self.forward_propagation(inputs[mu], w)
            val = np.sum(np.abs(expected_results[mu] - np.sign(output))/2)
            sum_val += val
        return sum_val

    def output_function(self, inputs : np.array, w : np.array):
        outputs = np.zeros((len(inputs), self.layer_sizes[-1]-1))
        for i, x in enumerate(inputs):
            outputs[i] = self.forward_propagation(x, w)
        return outputs

    def discrete_error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        p, _ = inputs.shape
        sum_val = 0
        for mu in range(p):
            output = self.forward_propagation(inputs[mu], w)
            val = np.sum(np.abs(expected_results[mu] - np.sign(output))/2)
            if val > 1:
                return 1500
            sum_val += val
        return sum_val

    def mse_error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        p, _ = inputs.shape
        val = 0
        for mu in range(p):
            output = self.forward_propagation(inputs[mu], w)
            val += 0.5 * np.sum((expected_results[mu] - output)**2)
        return val

    def get_encoder(self, w : np.array):
        return VAENetwork([x-1 for x in self.layer_sizes[:len(self.layer_sizes)//2]], self.activation_function, self.deriv_activation_function, self.error_type, self.interval, f"{self.title} encoder"), w[:len(self.layer_sizes)//2+1]

    def get_decoder(self, w : np.array):
        return VAENetwork([x-1 for x in self.layer_sizes[len(self.layer_sizes)//2:]], self.activation_function, self.deriv_activation_function, self.error_type, self.interval, f"{self.title} decoder"), w[len(self.layer_sizes)//2:]

    def export_weights(self, w : np.array, filename : str):
        with open(filename, 'w+') as f:
            f.write(f"{self.layer_sizes} {self.title}, {self.interval}\n")
            for m in range(len(w)):
                for j in range(self.layer_sizes[m+1]):
                    f.write(" ".join(map(str, w[m][j])) + "\n")
        nw = self.import_weights(filename)
        assert np.allclose(w, nw)

    def import_weights(self, filename : str):
        w = np.zeros((len(self.layer_sizes)-1, self.network_width, self.network_width))
        with open(filename, 'r') as f:
            f.readline()
            for m in range(len(self.layer_sizes)-1):
                for j in range(self.layer_sizes[m+1]):
                    w[m][j] = list(map(float, f.readline().split()))
        return w

    # def plot_evaluation(self, inputs, w):
    #     for i in inputs:
    #         values, _ = self._forward_propagation(i, w)
    #         plot_neural_network(self.layer_sizes, w, values)
