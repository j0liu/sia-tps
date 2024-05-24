import numpy as np

class OjaNetwork():

    def __init__(self, learning_rate_update):
        self.learning_rate_update = learning_rate_update
    
    def train_network(self, config, standarized_data, weights):
        w_hist = [weights.copy()]
        learning_rate = config['learning_rate']
        for epoch in range(config['epochs']):
            for x in standarized_data:
                activation = np.dot(x, weights)
                w_delta = learning_rate * activation * (x - activation * weights)
                weights += w_delta
                learning_rate = self.learning_rate_update(epoch)
            w_hist.append(weights.copy())
        return w_hist
        
