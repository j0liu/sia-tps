import numpy as np
from kohonen import KohonenNetwork
import json
import csv
from plotters import plot_heatmap
from oja import OjaNetwork
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotters import plot_first_principal_component, plot_energies, plot_patterns_over_time, plot_biplot, plot_boxplots
from hopfield import HopfieldNetwork
import pandas as pd
from letters import get_letter, add_noise, plot_all_patterns_together, export_pattern

def ej1_1_kohonen():
    with open('tp4/config/ej1-kohonen.json', 'r') as f:
        config = json.load(f)

    euclidean_distance = lambda x, y: np.sum((x - y)**2)**0.5
    learning_rate_update_function = lambda epoch: config['learning_rate'] / (1 + epoch * 0.1)
    # learning_rate_update_function = lambda epoch: 0.01
    radius_update_function = lambda epoch: config['radius']

    network = KohonenNetwork(output_size=config['k'], similarity_function=euclidean_distance, radius_update_function=radius_update_function, learning_rate_update_function=learning_rate_update_function)

    with open("tp4/europe.csv") as f:
        raw_data = list(csv.reader(f)) 
        variable_names = raw_data[0][1:]
        names = np.array(raw_data[1:])[:,0]
        nostd_data = np.array(np.array(raw_data[1:])[:,1:], dtype=float)
        data = StandardScaler().fit_transform(nostd_data)

    
    w_hist = network.train(config, network.initialize_weights(config, data), data)

    hits = np.zeros((network.output_size, network.output_size))
    hits_names = [["" for _ in range(network.output_size)] for _ in range(network.output_size)]

    weights = w_hist[-1]
    for idx, x in enumerate(data):
        winner_idx = np.argmin([network.similarity_function(x, w) for w in weights])
        winner_x, winner_y = divmod(winner_idx, network.output_size)
        hits[winner_x][winner_y] += 1
        if hits_names[winner_x][winner_y] == "":
            hits_names[winner_x][winner_y] = names[idx]
        else:
            hits_names[winner_x][winner_y] += "\n" + names[idx]
    
    avg_distances = network.get_distance_matrix(weights)

    plot_heatmap(hits, hits_names, 'Entries amount', 'Final entries per neuron')
    plot_heatmap(avg_distances, np.round(avg_distances, 3), 'Average distance', 'Average distance per neuron')
    #todo analisis por variable
    k = config['k']
    for i in range(len(variable_names)):
        plot_heatmap(weights[:,i].reshape(k,k), np.round(weights[:,i].reshape(k,k), 3), f'Variable {variable_names[i]}', 'Variable value per neuron')
    print(w_hist[-1])

def ej1_2_oja():
    with open('tp4/config/ej1.2-oja.json', 'r') as f:
        config = json.load(f)

    with open("tp4/europe.csv") as f:
        raw_data = list(csv.reader(f)) 
        variable_names = raw_data[0][1:]
        names = np.array(raw_data[1:])[:,0]
        nostd_data = np.array(np.array(raw_data[1:])[:,1:], dtype=float)
        standarized_data = StandardScaler().fit_transform(nostd_data)

    # Create box plots
    plot_boxplots(nostd_data, standarized_data, variable_names)
    
    network = OjaNetwork(lambda epoch: config['learning_rate'] / (1 + epoch))
    weights0 = np.random.uniform(0, 1, len(standarized_data[0]))

    w_hist = network.train_network(config, standarized_data, weights0)
    weights = w_hist[-1]

    # With library
    pca = PCA()
    pca_components = pca.fit_transform(standarized_data)

    print("y1 = " + " + ".join([f"{v} * {w:.3}" for v,w in zip(variable_names, weights)]))
        
    plot_first_principal_component(pca_components[:, 0], names, "PCA with library")
    plot_first_principal_component(np.dot(standarized_data, weights), names, "PCA with Oja")


    

def ej2_hopfield():
    with open('tp4/config/ej2-hopfield.json', 'r') as f:
        config = json.load(f)

    # Definición de patrones
    patterns = np.array([get_letter(letter) for letter in config['letters']])        

    # Creación de la red Hopfield
    hopfield_net = HopfieldNetwork(patterns)

    # Patrón ruidoso (una perturbación aleatoria del patrón A)
    targets = config['targets']

    for i,t in enumerate(targets):
        noisy_pattern = add_noise(t, config['noise_level'])
        if config.get("negate", False):
            noisy_pattern = -noisy_pattern

        patterns_over_time = hopfield_net.run(config, noisy_pattern)
        result = patterns_over_time[-1]
        # Cálculo de energía para cada patrón en el tiempo
        energies = [hopfield_net.energy(p) for p in patterns_over_time]
        
        # Graficar energía
        plot_energies(energies, title=f'{t}_{i}_energies')
        
        # Graficar patrones recuperados
        plot_patterns_over_time(config, patterns_over_time, title=f'{t}_{i}_process')

        if not any([np.array_equal(result, p) for p in patterns]):
            export_pattern(result)


if __name__ == '__main__':
    ej1_2_oja()