# read font.h and parse the letters

from multilayer import MultiLayerNetwork, hypercube_layers
import numpy as np
import activation_functions as af
import json

with open("tp5/config/linear_autoencoder.json") as f:
    config = json.load(f)

FILE_NAME = "tp5/font.txt"

def main():
  #read  the file
  with open(FILE_NAME, "r") as file:
    lines = file.readlines()
    inputs = []
    for line in lines:
      x = np.array([int(i) for i in line.split(' ')[1]])
      inputs.append(x)
    inputs = np.array(inputs)
    input_len = len(inputs[0])
    
    layer_sizes = [input_len,2, config['latent_space_dim'], 2,input_len]

    network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), "autoencoder")

if __name__ == "__main__":
  main()