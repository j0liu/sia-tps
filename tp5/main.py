# read font.h and parse the letters

from multilayer import MultiLayerNetwork, hypercube_layers
import numpy as np
import activation_functions as af
import json
from datetime import datetime
from plots import plot_comparison, plot_latent_space

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
  labels = labels[:len(inputs)]
  input_len = len(inputs[0])

  layer_sizes = [input_len, *ENCODER_LAYERS, config['latent_space_dim'], *(ENCODER_LAYERS[::-1]), input_len]

  network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1, 1), "autoencoder")
  norm_inputs = network.normalize(inputs.copy(), 0, 1)  
  denormalize = network.gen_denormalize_function(0, 1)

  if config['import']:
    w = network.import_weights(config['import_path'])
  else:
    t = datetime.now()
    w, _ = network.train_function(config, norm_inputs, norm_inputs)
    print(datetime.now() - t)
    network.export_weights(w, config['import_path'])

  # network.denormalized_error(norm_inputs, norm_inputs, w, denormalize)
  for i, (x, x2, l) in enumerate(zip(norm_inputs, network.output_function(norm_inputs, w), labels)):
    # print(i,x)
    plot_comparison(denormalize(x), denormalize(x2), f'{l}')
  
  encoder, w_encoder = network.get_encoder(w)
  plot_latent_space(encoder.output_function(norm_inputs, w_encoder), labels, "Latent space")




if __name__ == "__main__":
  main()