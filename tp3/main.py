import numpy as np
import sys
from functools import partial
import json
from perceptron import train_perceptron
import csv



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

def ejercicio_2():
    with open("tp3/data.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:], dtype=float)
    
    inputs = np.concatenate((np.ones((data.shape[0], 1)), data[:, :-1]), axis=1)
    outputs = data[:, -1]

    def linear_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
        p, dim = inputs.shape # p puntos en el plano, dim dimensiones
        o = lambda x: activation_function(np.dot(x, w))
        val = 0.5 * sum((expected[mu] - o(inputs[mu]))**2 for mu in range(p))
        return val
    
    # w = train_perceptron(config, inputs, outputs, lambda x: x, linear_error)
    # print(w)

    # print(f"x3 = {-w[2]/w[3]}*x2 + {-w[1]/w[3]}*x1 + {-w[0]/w[3]}")

    k_fold_cross_validation(config, np.concatenate((inputs, outputs.reshape(-1, 1)), axis=1), lambda x: x, linear_error)

    

def k_fold_cross_validation(config, inputs, activation_function, error_function, deriv_activation_function = lambda x: 1):
    np.random.shuffle(inputs)
    k = config["k"]
    p, dim = inputs.shape
    fold_size = p // k
    errors = []
    for i in range(k):
        train = np.concatenate((inputs[:i*fold_size], inputs[(i+1)*fold_size:]), axis=0)
        test = inputs[i*fold_size:(i+1)*fold_size]
        w = train_perceptron(config, train[:, :-1], train[:, -1], activation_function, error_function, deriv_activation_function)
        errors.append((w, error_function(test[:, :-1], test[:, -1], w, activation_function)))
    # print(errors)
    (w_min, err_min) = min(errors, key=lambda x: abs(x[1]))
    print(f"[{','.join(str(f) for f in w_min.tolist())}]")
    print(err_min)
    print(f"x3 = {-w_min[2]/w_min[3]}*x2 + {-w_min[1]/w_min[3]}*x1 + {-w_min[0]/w_min[3]}")
    # return sum(errors) / k 



if __name__ == "__main__":
    ejercicio_2()
