import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class HopfieldNetwork:
    
    def __init__(self, patterns):
        self.patterns = patterns
        self.num_neurons = patterns.shape[1]
        self.weights = self._initialize_weights(patterns)
    
    def _initialize_weights(self, patterns):
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
    
    def run(self, config, initial_pattern):
        current_pattern = initial_pattern.copy()
        history = [current_pattern.copy()]
        
        for step in range(config['steps']):
            # Asynchronous update
            for i in np.random.permutation(self.num_neurons):
                current_pattern[i] = np.sign(np.dot(self.weights[i], current_pattern))
            
            history.append(current_pattern.copy())
        
        return history
