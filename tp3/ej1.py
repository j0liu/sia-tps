import json

import numpy as np
from perceptron import train_perceptron

with open("tp3/config.json") as f:
    config = json.load(f)


def ejercicio_1():
    inputs = np.array([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    expected_and = np.array([-1, -1, -1, 1])
    expected_or = np.array([1, 1, -1, -1])

    step_function = lambda x: 1 if x >= 0 else -1

    def step_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
        o = lambda x: activation_function(np.dot(x, w))
        return sum(o(inputs[mu]) != expected[mu] for mu in range(len(expected))) / len(expected)

    w_and = train_perceptron(config, inputs, expected_and, step_function, step_error)
    w_or  = train_perceptron(config, inputs, expected_or, step_function, step_error)
    print(w_and)
    print(w_or)

    print(f"x2 = {-w_and[1]/w_and[2]}*x1 + {-w_and[0]/w_and[2]}")
    print(f"x2 = {-w_or[1]/w_or[2]}*x1 + {-w_or[0]/w_or[2]}")