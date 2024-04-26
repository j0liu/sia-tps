import numpy as np
import sys
from functools import partial


#puntos en el plano: n = 2
#inputs p x n , expected p x 1
#activation_function: theta
#expected: tzeta
#se debe tomar epsilon 0 para step
def train_perceptron(config : dict, inputs : np.matrix, expected : np.array, activation_function, error_function, deriv_activation_function = lambda x: 1):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones

    i = 0
    w = np.concatenate((np.array([config['bias']]), (np.random.rand(dim-1)))) # pesos
    error = None
    min_error = sys.maxsize
    w_min = None
    while min_error > config['epsilon'] and i < config['limit']:
        mu = np.random.randint(0, p)
        h = np.dot(inputs[mu],w)
        o = activation_function(h) 

        delta_w = config['learning_rate'] * (expected[mu] - o) * inputs[mu] * deriv_activation_function(h)
        w += delta_w
        # error_o = 0.5 * np.sum((expected[mu] - o[mu]) ** 2)
        error = error_function(inputs, expected, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = w
        i += 1
    return w_min

