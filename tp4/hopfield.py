import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import letters

class HopfieldNetwork:
    
    def __init__(self, patterns):
        self.patterns = patterns
        self.num_neurons = patterns.shape[1]
        self.weights = self.initialize_weights(patterns)
    
    def initialize_weights(self, patterns):
        n = self.num_neurons
        W = np.zeros((n, n))
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0)
        return W / n
    
    def update(self, pattern):
        return np.sign(np.dot(self.weights, pattern))
    
    def energy(self, pattern):
        return -0.5 * np.dot(pattern.T, np.dot(self.weights, pattern))
    
    def run(self, config, input):
        current_pattern = input.copy()
        history = [current_pattern.copy()]
        
        for _ in range(config['steps']):
            # Asynchronous update
            # for i in np.random.permutation(self.num_neurons):
                # current_pattern[i] = np.sign(np.dot(self.weights[i], current_pattern))
            current_pattern = np.sign(np.inner(self.weights, current_pattern))
            history.append(current_pattern.copy())
            if any([np.array_equal(current_pattern, p) for p in history[:-1]]):
                break
        
        return history
