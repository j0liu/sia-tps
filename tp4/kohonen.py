import numpy as np

def squared_neighborhood(output_size, i, radius):
    neighbors = []
    x = i % output_size
    y = i // output_size
    pos = np.array([x, y])
    neighbors = []
    for j in range(output_size**2):
        pos2 = np.array([j % output_size, j // output_size])
        if np.linalg.norm(pos - pos2) <= radius:
            neighbors.append(j)
    return neighbors
    


class KohonenNetwork():
    def __init__(self, output_size: int, similarity_function, radius_update_function, learning_rate_update_function):
        self.output_size = output_size
        self.similarity_function = similarity_function
        self.radius_update_function = radius_update_function
        self.learning_rate_update_function = learning_rate_update_function
        pass

    def initialize_weights(self, config : dict, inputs : np.array):
        input_dim = len(inputs[0])
        k2 = self.output_size**2
        if config.get('random_start', True):
            return np.random.rand(k2, input_dim)
        else:
            return np.random.choice(inputs, k2)


    def train(self, config : dict, weights : np.array, inputs : np.array):
        weight_history = [weights.copy()]
        for epoch in range(config['limit']):
            learning_rate = self.learning_rate_update_function(epoch)
            radius = self.radius_update_function(epoch)


            x = inputs[np.random.choice(len(inputs))]
            winner_idx = np.argmin([self.similarity_function(x, w) for w in weights])

            neighbors = squared_neighborhood(self.output_size, winner_idx, radius)

            for n in neighbors:
                weights[n] += learning_rate * (x - weights[n])

            weight_history.append(weights.copy())

        return weight_history
