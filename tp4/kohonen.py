import numpy as np

def squared_neighborhood(output_size, i, radius):
    neighbors = []
    x,y = divmod(i, output_size)
    pos = np.array([x, y])
    neighbors = []
    for j in range(output_size**2):
        # pos2 = np.array([j // output_size, j % output_size])
        pos2 = np.array([*divmod(j, output_size)])
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
    
    def get_distance_matrix(self,weights : np.array):
        distance_matrix = np.zeros((self.output_size, self.output_size))
        for i in range(self.output_size):
            for j in range(self.output_size):
                idx = i*self.output_size+j
                neighborhood = squared_neighborhood(self.output_size, idx, 1)
                neighborhood.remove(idx)
                neighborhood_distances = np.array([np.linalg.norm(weights[n] - weights[idx]) for n in neighborhood])
                distance_matrix[i][j] = np.mean(neighborhood_distances)
        return distance_matrix
