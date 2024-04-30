import numpy as np
import sys
from functools import partial
import json
from perceptron import train_perceptron, simple_error
import csv
import activation_functions as af

with open("tp3/config.json") as f:
    config = json.load(f)

def ejercicio_2():
    with open("tp3/data.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:], dtype=float)
    
    inputs = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)

    beta = config["beta"]
    
    print("linear")
    # Linear
    linear_results = k_fold_cross_validation(config, inputs, af.id, simple_error, deriv_activation_function=af.one)
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
    train_errors = []
    for (w, test, train) in results:
        errors.append(error_function(test[:, :-1], denormalize_function(test[:, -1]), w, lambda x: denormalize_function(activation_function(x))))
        train_errors.append(error_function(train[:, :-1], denormalize_function(train[:, -1]), w, lambda x: denormalize_function(activation_function(x))))
        print(w)
    print(f"Error medio con test: {np.mean(errors)}")
    print(f"Error medio con train: {np.mean(train)}")


def k_fold_cross_validation(config, inputs, activation_function, error_function, deriv_activation_function = lambda x: 1):
    np.random.shuffle(inputs)
    k = config["k"]
    p, dim = inputs.shape
    fold_size = p // k
    results = []
    for i in range(k):
        test = inputs[i*fold_size:(i+1)*fold_size]
        train = np.concatenate((inputs[:i*fold_size], inputs[(i+1)*fold_size:]), axis=0)
        w = train_perceptron(config, train[:, :-1], train[:, -1], activation_function, error_function, deriv_activation_function)
        results.append((w, test, train))

    return results

if __name__ == "__main__":
    ejercicio_2()
