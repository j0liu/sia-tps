import json

import numpy as np
from single_layer import step_error
import activation_functions as af
from single_layer import SingleLayerNetwork, step_error

with open("tp3/config/ej1.json") as f:
    config = json.load(f)


def ejercicio_1():
    inputs = np.array([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    expected_and = np.array([-1, -1, -1, 1])
    expected_or = np.array([1, 1, -1, -1])

    network = SingleLayerNetwork(af.step, af.one, step_error)

    w_and, _ = network.train_function(config, inputs, expected_and, 'AND')
    w_xor, _  = network.train_function(config, inputs, expected_or, 'XOR')
    print(w_and)
    print(w_xor)

    print(f"x2 = {-w_and[1]/w_and[2]}*x1 + {-w_and[0]/w_and[2]}")
    print(f"x2 = {-w_xor[1]/w_xor[2]}*x1 + {-w_xor[0]/w_xor[2]}")


if __name__ == "__main__":
    ejercicio_1()