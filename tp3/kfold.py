import numpy as np
from functools import partial
from metrics import accuracy, precision, recall, f1_score
from plotTrainTest import plot_k_fold_errors, plot_metrics
import activation_functions as af
from network_abc import NetworkABC

def analyze_method(config : dict, inputs : np.array, expected: np.array, network: NetworkABC, min_expected, max_expected):
    expected = network.normalize(expected, min_expected, max_expected) # Normalization
    denormalize = network.gen_denormalize_function(min_expected, max_expected)
    results = k_fold_cross_validation(config, inputs, expected, network)
    process_k_fold_cross_validation_results(results, network, denormalize)


def analyze_method_categorization(config : dict, inputs : np.array, expected: np.array, network: NetworkABC, min_expected, max_expected):
    expected = network.normalize(expected, min_expected, max_expected)
    denormalize = network.gen_denormalize_function(min_expected, max_expected)
    results = k_fold_cross_validation(config, inputs, expected, network)
    process_k_fold_cross_categorization_results(results, network, denormalize)
    return results

def process_k_fold_cross_validation_results(results, network: NetworkABC, denormalize_function = lambda x: x):
    errors = []
    train_errors = []
    for (w, test, expected_test, train, expected_train, w_hist) in results:        
        # errors.append(network.denormalized_error(test, expected_test, w))
        # train_errors.append(network.denormalized_error(test, expected_train, w))
        errors.append(network.denormalized_error(test, expected_test, w, denormalize_function)/len(test))
        train_errors.append(network.denormalized_error(train, expected_train, w, denormalize_function)/len(train))

    # get index of min value in errors

    # min_error_index = np.argmin(errors)
    min_error_index = np.argmin(train_errors)
    (_, min_test, min_expected_test, min_train, min_expected_train, min_w_hist) = results[min_error_index]
    train_e_list = [network.denormalized_error(min_train, min_expected_train, w, denormalize_function)/len(min_train) for w in min_w_hist]
    test_e_list = [network.denormalized_error(min_test, min_expected_test, w, denormalize_function)/len(min_test) for w in min_w_hist]
    
    plot_k_fold_errors(test_e_list, train_e_list, network.title)


    print(f"Error minimo con test ({len(min_test)}): {errors[min_error_index]}")
    print(f"Error minimo con train ({len(min_train)}): {train_errors[min_error_index]}")

    
def process_k_fold_cross_categorization_results(results, network: NetworkABC, denormalize_function = lambda x : x):
    
    metrics = []
    for i, (w, test, expected_test, train, expected_train, _) in enumerate(results):
        train_outputs = denormalize_function(network.output_function(train, w))
        test_outputs = denormalize_function(network.output_function(test, w))
        train_confusion_matrix = get_confusion_matrix(train_outputs, denormalize_function(expected_train))
        test_confusion_matrix = get_confusion_matrix(test_outputs, denormalize_function(expected_test))
        metrics.append(accuracy(test_confusion_matrix) + precision(test_confusion_matrix) + recall(test_confusion_matrix) + f1_score(test_confusion_matrix))
        # print(f"Pesos: {w}")
        print(f"w{i}:")
        print(f"train {len(train)}: Acurracy: {accuracy(train_confusion_matrix)} Precision: {precision(train_confusion_matrix)} Recall: {recall(train_confusion_matrix)} F1 Score: {f1_score(train_confusion_matrix)}")
        print(f"test {len(test)}: Acurracy: {accuracy(test_confusion_matrix)} Precision: {precision(test_confusion_matrix)} Recall: {recall(test_confusion_matrix)} F1 Score: {f1_score(test_confusion_matrix)}")
        # network.export_weights(w, f"tp3/weights/weights_{i}.txt")

    # Select the best 
    best_index = np.argmax(metrics)
    metrics_to_plot = []
    for w in results[best_index][5]:
        train_outputs = denormalize_function(network.output_function(results[best_index][3], w))
        test_outputs = denormalize_function(network.output_function(results[best_index][1], w))
        train_confusion_matrix = get_confusion_matrix(train_outputs, denormalize_function(results[best_index][4]))
        test_confusion_matrix = get_confusion_matrix(test_outputs, denormalize_function(results[best_index][2]))
        metrics_to_plot.append([accuracy(test_confusion_matrix), precision(test_confusion_matrix), recall(test_confusion_matrix), f1_score(test_confusion_matrix), accuracy(train_confusion_matrix), precision(train_confusion_matrix), recall(train_confusion_matrix), f1_score(train_confusion_matrix)])
    
    plot_metrics(metrics_to_plot, network.title)


        



def get_confusion_matrix(outputs, expected_list):
    confusion_matrix = np.zeros((2, 2))
    ones = np.ones(len(outputs[0]))
    for i in range(len(outputs)):
        output = np.round(outputs[i]) # Round to 0 or 1
        expected = np.round(expected_list[i])

        confusion_matrix[1][1] += np.sum((ones - output) * (ones - expected)) #tn
        confusion_matrix[1][0] += np.sum((ones - output) * expected) #fp
        confusion_matrix[0][1] += np.sum(output * (ones - expected)) #fn
        confusion_matrix[0][0] += np.sum(expected * output) #tp
    return confusion_matrix


    
def k_fold_cross_validation(config, inputs, expected, network: NetworkABC):
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
        w, w_hist= network.train_function(config, train, train_expected)
        results.append((w, test, test_expected, train, train_expected, w_hist))

    return results