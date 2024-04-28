import numpy as np
import sys
from functools import partial
import json
from perceptron import train_perceptron
import csv
import activation_functions as af


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


def simple_error(inputs : np.array, expected : np.array, w : np.array, activation_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones
    o = lambda x: activation_function(np.dot(x, w))
    val = 0.5 * sum((expected[mu] - o(inputs[mu]))**2 for mu in range(p))
    return val

def ejercicio_2():
    with open("tp3/data.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:], dtype=float)
    
    inputs = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    # outputs = data[:, -1]

    
    # w = train_perceptron(config, inputs, outputs, lambda x: x, simple_error)
    # print(w)  
    beta = config["beta"]
    
    print("linear")
    # Linear
    linear_results = k_fold_cross_validation(config, inputs, af.id, simple_error, af.one)
    process_k_fold_cross_validation_results(linear_results, af.id, af.id, simple_error)

    print("tanh")
    # Non linear - tanh
    analyze_simple_method(config, np.copy(inputs), lambda x: af.tanh(x, beta), lambda x: af.tanh_derivative(x, beta), -1, 1)

    # Non linear - logistic
    print("logistic")
    analyze_simple_method(config, np.copy(inputs), lambda x: af.logistic(x, beta), lambda x: af.logistic_derivative(x, beta), 0, 1)
    

def analyze_simple_method(config : dict, inputs : np.array, activation_function, derivative_function, interval_min, interval_max):
    min_expected = min(inputs[:, -1])
    max_expected = max(inputs[:, -1])
    inputs[:, -1] = af.normalize(inputs[:, -1], min_expected, max_expected, interval_min, interval_max) # Normalization

    denormalize = partial(af.denormalize, x_min=min_expected, x_max=max_expected, a=interval_min, b=interval_max)
    results = k_fold_cross_validation(config, inputs, activation_function, simple_error, derivative_function)
    process_k_fold_cross_validation_results(results, activation_function, denormalize, simple_error)


def process_k_fold_cross_validation_results(results, activation_function, denormalize_function, error_function):
    errors = []
    for (w, test) in results:
        errors.append(error_function(test[:, :-1], denormalize_function(test[:, -1]), w, lambda x: denormalize_function(activation_function(x))))
        print(w)
    print(errors)


def k_fold_cross_validation(config, inputs, activation_function, error_function, deriv_activation_function = lambda x: 1):
    np.random.shuffle(inputs)
    k = config["k"]
    p, dim = inputs.shape
    fold_size = p // k
    # train_errors = []
    # test_errors = []
    results = []
    for i in range(k):
        test = inputs[i*fold_size:(i+1)*fold_size]
        train = np.concatenate((inputs[:i*fold_size], inputs[(i+1)*fold_size:]), axis=0)
        w = train_perceptron(config, train[:, :-1], train[:, -1], activation_function, error_function, deriv_activation_function)
        results.append((w, test))
        # test_errors.append((w, error_function(test[:, :-1], test[:, -1], w, activation_function)))
        # train_errors.append((w, error_function(train[:, :-1], train[:, -1], w, activation_function)))

    return results
    # print(errors)
    # (w_min, err_min) = min(test_errors, key=lambda x: abs(x[1]))
    # print("test")
    # [print(e) for e in test_errors]
    # print("train")
    # [print(e) for e in train_errors]
    # print(f"[{','.join(str(f) for f in w_min.tolist())}]")
    # print(err_min)
    # print(f"x3 = {-w_min[2]/w_min[3]}*x2 + {-w_min[1]/w_min[3]}*x1 + {-w_min[0]/w_min[3]}")
    # return sum(errors) / k 


if __name__ == "__main__":
    ejercicio_2()
