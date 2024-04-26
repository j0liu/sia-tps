import numpy as np
import sys
from functools import partial



LIMIT = 100
LEARNING_RATE = 0.1
BIAS = 0

def output(x: np.array, w: np.array, activation_function):
    return activation_function(np.dot(x, w))

def linear_error(inputs : np.matrix, expected : np.array, w : np.array, activation_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones

    return 0.5 * sum(expected[mu] - activation_function(sum(np.dot(np.ravel(inputs[mu]),w[i]) for i in range(dim))) for mu in range(p))

def step_error(inputs : np.matrix, expected : np.array, w : np.array, activation_function):
    o = partial(output, w=w, activation_function=activation_function)
    return sum(o(np.ravel(inputs[mu])) != expected[mu] for mu in range(len(expected))) / len(expected)


#puntos en el plano: n = 2
#inputs p x n , expected p x 1
#activation_function: theta
#expected: tzeta
def train_perceptron(inputs : np.matrix, expected : np.array, bias: float, activation_function, error_function):
    p, dim = inputs.shape # p puntos en el plano, dim dimensiones

    i = 0
    w = np.concatenate((np.array([bias]), (np.random.rand(dim-1)))) # pesos
    error = None
    min_error = sys.maxsize
    w_min = None
    while min_error > 0 and i < LIMIT:
        mu = np.random.randint(0, p)
        o = output(np.ravel(inputs[mu]), w, activation_function)
        delta_w = LEARNING_RATE * (expected[mu] - o) * np.ravel(inputs[mu])
        w += delta_w
        # error_o = 0.5 * np.sum((expected[mu] - o[mu]) ** 2)
        error = error_function(inputs, expected, w, activation_function)
        if error < min_error:
            min_error = error
            w_min = w
        i += 1
    return w_min


def main():
    inputs = np.matrix([[1, -1, 1], [1, 1, -1], [1, -1, -1], [1, 1, 1]])
    expected_and = np.array([-1, -1, -1, 1])
    expected_or = np.array([1, 1, -1, -1])

    step_function = lambda x: 1 if x >= 0 else -1

    w_and = train_perceptron(inputs, expected_and, BIAS, step_function, step_error)
    w_or  = train_perceptron(inputs, expected_or, BIAS, step_function, step_error)
    print(w_and)
    print(w_or)

    print(f"x2 = {-w_and[1]/w_and[2]}*x1 + {-w_and[0]/w_and[2]}")
    print(f"x2 = {-w_or[1]/w_or[2]}*x1 + {-w_or[0]/w_or[2]}")

if __name__ == "__main__":
    main()
