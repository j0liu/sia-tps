import numpy as np
from kohonen import KohonenNetwork
import json
import csv
from plotters import plot_heatmap

def ej1_kohonen():
    with open('tp4/config/ej1-kohonen.json', 'r') as f:
        config = json.load(f)

    euclidean_distance = lambda x, y: np.sum((x - y)**2)**0.5
    learning_rate_update_function = lambda epoch: config['learning_rate'] / (1 + epoch)
    radius_update_function = lambda epoch: config['radius']

    network = KohonenNetwork(output_size=config['k'], similarity_function=euclidean_distance, radius_update_function=radius_update_function, learning_rate_update_function=learning_rate_update_function)

    with open("tp4/europe.csv") as f:
        data = list(csv.reader(f)) 
        names = np.array(data[1:])[:,0]
        data = np.array(data[1:])[:,1:]
        data = np.array(data[:,1:], dtype=float)

    w_hist = network.train(config, network.initialize_weights(config, data), data)

    hits = np.zeros((network.output_size, network.output_size))
    hits_names = [["" for _ in range(network.output_size)] for _ in range(network.output_size)]

    for idx, x in enumerate(data):
        winner_idx = np.argmin([network.similarity_function(x, w) for w in w_hist[-1]])
        winner_x, winner_y = divmod(winner_idx, network.output_size)
        hits[winner_x][winner_y] += 1
        if hits_names[winner_x][winner_y] == "":
            hits_names[winner_x][winner_y] = names[idx]
        else:
            hits_names[winner_x][winner_y] += "\n" + names[idx]

    plot_heatmap(hits, hits_names)
    print(w_hist[-1])
if __name__ == '__main__':
    ej1_kohonen()