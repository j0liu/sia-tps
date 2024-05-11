import numpy as np
from kohonen import KohonenNetwork
import json
import csv

def ej1_kohonen():
    with open('tp4/config/ej1-kohonen.json', 'r') as f:
        config = json.load(f)

    euclidean_distance = lambda x, y: np.sum((x - y)**2)**0.5
    learning_rate_update_function = lambda epoch: config['learning_rate'] / (1 + epoch)
    radius_update_function = lambda epoch: config['radius']

    network = KohonenNetwork(output_size=config['k'], similarity_function=euclidean_distance, radius_update_function=radius_update_function, learning_rate_update_function=learning_rate_update_function)

    with open("tp4/europe.csv") as f:
        data = list(csv.reader(f)) 
        data = np.array(data[1:])[:,1:]
        names = data[:,0]
        data = np.array(data[:,1:], dtype=float)

    w_hist = network.train(config, network.initialize_weights(config, data), data)
    print(w_hist[-1])


if __name__ == '__main__':
    ej1_kohonen()