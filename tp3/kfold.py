import numpy as np
from functools import partial
from perceptron import train_perceptron
from plotTrainTest import plot_k_fold_errors
import activation_functions as af

def analyze_method(config : dict, inputs : np.array, expected: np.array, activation_function, derivative_function, interval_min, interval_max, error_function, title):
    min_expected = min(expected)
    max_expected = max(expected)
    expected = af.normalize(expected, min_expected, max_expected, interval_min, interval_max) # Normalization

    denormalize = partial(af.denormalize, x_min=min_expected, x_max=max_expected, a=interval_min, b=interval_max)
    results = k_fold_cross_validation(config, train_perceptron,
                                      inputs, expected, activation_function, error_function, title, derivative_function)
    process_k_fold_cross_validation_results(results, activation_function, denormalize, error_function, title)


def process_k_fold_cross_validation_results(results, activation_function, denormalize_function, error_function, title):
    errors = []
    train_errors = []
    for (w, test, expected_test, train, expected_train) in results:
        errors.append(error_function(test, denormalize_function(expected_test), w, lambda x: denormalize_function(activation_function(x))))
        train_errors.append(error_function(train, denormalize_function(expected_train), w, lambda x: denormalize_function(activation_function(x))))
        print(w)
    plot_k_fold_errors(errors, train_errors, title)
    print(f"Error medio con test: {np.mean(errors)}")
    print(f"Error medio con train: {np.mean(train)}")


def k_fold_cross_validation(config, train_perceptron_function, inputs, expected, activation_function, error_function, title, deriv_activation_function = lambda x: 1):
    # shuffle inputs with their corresponding expected values
    inputs_and_expected = list(zip(inputs, expected))
    np.random.shuffle(inputs_and_expected)
    inputs = np.array([x[0] for x in inputs_and_expected])
    expected = np.array([x[1] for x in inputs_and_expected])

    k = config["k"]
    p, dim = inputs.shape
    fold_size = p // k
    results = []
    for i in range(k):
        test = inputs[i*fold_size:(i+1)*fold_size]
        test_expected = expected[i*fold_size:(i+1)*fold_size]
        train = np.concatenate((inputs[:i*fold_size], inputs[(i+1)*fold_size:]), axis=0)
        train_expected = np.concatenate((expected[:i*fold_size], expected[(i+1)*fold_size:]), axis=0)
        w = train_perceptron_function(config, train, train_expected, activation_function, title, error_function, deriv_activation_function)
        results.append((w, test, test_expected, train, train_expected))

    return results