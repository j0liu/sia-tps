import numpy as np
import sys
from functools import partial
from plotSimplePerceptron import plot_decision_boundary, plot_errors
from network_abc import NetworkABC

def simple_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones
    o = lambda x: activation_function(np.dot(x, w))
    val = 0.5 * sum((expected[mu] - o(inputs[mu]))**2 for mu in range(p))
    return val

def step_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
    o = lambda x: activation_function(np.dot(x, w))
    error = sum(o(inputs[mu]) != expected[mu] for mu in range(len(expected))) / len(expected)
    return error

class SingleLayerNetwork(NetworkABC):
    def __init__(self, activation_function, deriv_activation_function, error_function, interval = None, title = ""):
        super().__init__(activation_function, deriv_activation_function, interval, title)
        self.base_error_function = error_function
        
    #puntos en el plano: n = 2
    #x p x n , expected p x 1
    #activation_function: theta
    #expected: tzeta
    #se debe tomar epsilon 0 para step
    def train_function(self, config : dict, inputs : np.array, expected_results : np.array):
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
            o = self.activation_function(h)

            delta_w = config['learning_rate'] * (expected_results[mu] - o) * self.deriv_activation_function(h) * inputs[mu]
            w += delta_w
            w_hist.append(w.copy())
            error = self.error_function(inputs, expected_results, w)
            if error < min_error:
                min_error = error
                w_min = w_hist[-1]
            i += 1
            e_list.append(error.copy())
        print("iteraciones:", i)
        # plot_decision_boundary(w_hist, inputs, expected_results, self.title)
        # plot_errors(e_list, self.title)
        return w_min, w_hist
    
    def output_function(self, inputs : np.array, w : np.array):
        raise 'Not implemented'


    def error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        return self.base_error_function(inputs, expected_results, w, self.activation_function)
    
    def export_weights(self, w : np.array, filename : str):
        with open(filename, 'w') as f:
            f.write(f"{self.title}, {self.interval}\n")
            for m in range(len(w)):
                for j in range(len(w[m])):
                    f.write(f"{w[m][j]}\n")

    def import_weights(self, filename : str):
        with open(filename, 'r') as f:
            f.readline()
            w = []
            for m in range(len(self.layer_sizes)-1):
                w.append([])
                for j in range(self.layer_sizes[m+1]):
                    w[m].append(list(map(float, f.readline().split())))
            return np.array(w)