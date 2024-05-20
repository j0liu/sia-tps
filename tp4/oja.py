import numpy as np

class OjaNetwork():

    def __init__(self, weights):
        self.weights = weights

    def update_weights(self, activation, learing_rate, inputs):
        inputs = np.array(inputs)
        activation = np.array(activation)
        w_0 = np.array(self.weights)
        w_delta = learing_rate * (inputs*activation - activation**2 * w_0)
        self.weights = w_0 + w_delta
        return self.weights
    
    def get_activation(self, inputs):
        inputs = np.array(inputs)
        return np.dot(inputs, self.weights)
    
    def train_network(self, config, standarized_data):
        w_hist = [self.weights.copy()]
        learning_rate = config['learning_rate']
        for epoch in range(config['epochs']):
            for x in standarized_data:
                activation = self.get_activation(x)
                self.update_weights(activation, learning_rate, x)
                learning_rate = learning_rate/(1+epoch)
            w_hist.append(self.weights.copy())
        return w_hist
        
