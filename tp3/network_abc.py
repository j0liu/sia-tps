from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Callable
import activation_functions as af

class NetworkABC(ABC):
    def __init__(self, activation_function: Callable[[float],float], deriv_activation_function: Callable[[float],float], interval: tuple[float, float] = None, title = ""):
        self.activation_function = activation_function
        self.deriv_activation_function = deriv_activation_function
        self.interval = interval
        self.title = title

    @abstractmethod
    def train_function(self, config : dict, inputs : np.array, expected_results : np.array):
        pass

    @abstractmethod
    def output_function(self, inputs : np.array, w : np.array) -> np.array:
        pass
        
    @abstractmethod
    def error_function(self, inputs : np.array, expected_results : np.array, w : np.array):
        pass

    def denormalized_error(self, inputs : np.array, expected_results : np.array, w : np.array, denormalize_function):
        aux = self.activation_function
        self.activation_function = lambda x: denormalize_function(aux(x))
        error = self.error_function(inputs, denormalize_function(expected_results), w)
        self.activation_function = aux
        return error
    
    def normalize(self, x, min_expected, max_expected):
        return af.scale(x, min_expected, max_expected, self.interval[0], self.interval[1])
    
    def gen_denormalize_function(self, min_expected, max_expected) -> Callable[[float],float]:
        return lambda x: af.scale(x, self.interval[0], self.interval[1], min_expected, max_expected)