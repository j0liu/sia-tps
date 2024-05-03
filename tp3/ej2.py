import numpy as np
import sys
from functools import partial
import json
from perceptron import train_perceptron, simple_error
import csv
import activation_functions as af
from kfold import k_fold_cross_validation, process_k_fold_cross_validation_results, analyze_method

with open("tp3/config.json") as f:
    config = json.load(f)

def ejercicio_2():
    with open("tp3/data.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:], dtype=float)
    
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    inputs = data[:,:-1]
    expected = data[:,-1]

    beta = config["beta"]
    
    print("linear")
    # Linear
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    linear_results = k_fold_cross_validation(config, train_perceptron, input_copy, expected_copy, af.id, simple_error, "linear", deriv_activation_function=af.one)
    process_k_fold_cross_validation_results(linear_results, af.id, af.id, simple_error, "linear")

    print("tanh")
    # Non linear - tanh
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    analyze_method(config, input_copy, expected_copy, af.gen_tanh(beta), af.gen_tanh_derivative(beta), -1, 1, simple_error, "tanh")

    # Non linear - logistic
    print("logistic")
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    analyze_method(config, input_copy, expected_copy, af.gen_logistic(beta), af.gen_logistic_derivative(beta), 0, 1, simple_error, "logistic")

if __name__ == "__main__":
    ejercicio_2()
