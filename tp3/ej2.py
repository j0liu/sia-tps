import numpy as np
import sys
from functools import partial
import json
from single_layer import simple_error, SingleLayerNetwork
import csv
import activation_functions as af
from kfold import k_fold_cross_validation, process_k_fold_cross_validation_results, analyze_method

from single_layer import SingleLayerNetwork

with open("tp3/config/ej2.json") as f:
    config = json.load(f)

def ejercicio_2():
    with open("tp3/data.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:], dtype=float)
    
    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    inputs = data[:,:-1]
    expected = data[:,-1]

    beta = config["beta"]

    # train_perceptron_function = partial(train_perceptron, error_function=simple_error)
    network = SingleLayerNetwork(af.id, af.one, simple_error, title="linear")
    
    print("linear")
    # Linear
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    linear_results = k_fold_cross_validation(config, input_copy, expected_copy, network)
    process_k_fold_cross_validation_results(linear_results, network)

    print("tanh")
    # Non linear - tanh
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    networkTan = SingleLayerNetwork(af.gen_tanh(beta), af.gen_tanh_derivative(beta), simple_error, (-1,1),title="tanh")
    analyze_method(config, input_copy, expected_copy, networkTan, min(expected_copy), max(expected_copy))
    
    # Non linear - logistic
    print("logistic")
    input_copy = np.copy(inputs)
    expected_copy = np.copy(expected)
    networkTan = SingleLayerNetwork(af.gen_logistic(beta), af.gen_logistic_derivative(beta), simple_error, (0,1), title="logistic")
    analyze_method(config, input_copy, expected_copy, networkTan, min(expected_copy), max(expected_copy))

if __name__ == "__main__":
    ejercicio_2()
