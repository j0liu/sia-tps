# read font.h and parse the letters

from multilayer import MultiLayerNetwork, hypercube_layers
import numpy as np
import activation_functions as af
import json
import matplotlib.pyplot as plt
from datetime import datetime

with open("tp5/config/linear_autoencoder.json") as f:
    config = json.load(f)

FILE_NAME = "tp5/font.txt"

ENCODER_LAYERS = [25, 25]

def main():
  #read  the file
  with open(FILE_NAME, "r") as file:
    lines = file.readlines()
  inputs = []
  labels = []
  for line in lines:
    x = np.array([int(i) for i in line.split(' ')[1].replace('\n', '')])
    inputs.append(x)
    labels.append(line.split(' ')[0])
  inputs = np.array(inputs)#[:-1]
  input_len = len(inputs[0])
  
  layer_sizes = [input_len, *ENCODER_LAYERS, config['latent_space_dim'], *(ENCODER_LAYERS[::-1]), input_len]

  network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1, 1), "autoencoder")
  norm_inputs = network.normalize(inputs.copy(), 0, 1)  
  denormalize = network.gen_denormalize_function(0, 1)

  t = datetime.now()
  w, w_hist = network.train_function(config, norm_inputs, norm_inputs)
  print(datetime.now() - t)

  network.denormalized_error(norm_inputs, norm_inputs, w, denormalize)
  for i, (x, x2, l) in enumerate(zip(norm_inputs, network.output_function(norm_inputs, w), labels)):
    # print(i,x)
    plot_single_pattern(denormalize(x), denormalize(x2), f'{l}')
  

# TODO: move to plots.py
def plot_single_pattern(original, reconstructed, label):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(-original.reshape(7, 5), cmap='summer')
    plt.title(f'Original {label}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(-reconstructed.reshape(7, 5), cmap='summer')
    plt.title(f'Reconstructed {label}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(-np.round(reconstructed.reshape(7, 5),0), cmap='summer')
    plt.title(f'Rounded {label}')
    plt.axis('off')
    plt.savefig(f"tp5/plots/{label}.png")
    # plt.show()
    plt.close()



if __name__ == "__main__":
  main()