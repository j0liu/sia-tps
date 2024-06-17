# read font.h and parse the letters

from multilayer import MultiLayerNetwork, hypercube_layers
import numpy as np
import activation_functions as af
import json
from datetime import datetime
from plots import plot_comparison, plot_latent_space


FILE_NAME = "tp5/font.txt"

def read_letters(file_name = FILE_NAME):
  with open(file_name, "r") as file:
    lines = file.readlines()
  inputs = []
  labels = []
  for line in lines:
    print(line.replace('\n', '').split(' ')[1:])
    x = np.array([float(i) for i in line.replace('\n', '').split(' ')[1:]])
    inputs.append(x)
    labels.append(line.split(' ')[0])
  inputs = np.array(inputs)#[:-1]
  labels = labels[:len(inputs)]
  return inputs, labels

def generate_noise(labels: str, inputs : np.array):
  with open(f"tp5/noisy_font.txt", 'w') as file:
    for i, num_pixels in enumerate(inputs):
      for lvl in [0.1, 0.25, 0.5]:
        noise = np.random.normal(0, lvl, num_pixels.shape)
        noisy_inputs = np.clip(num_pixels + noise, 0, 1)

        file.write(labels[i])
        for pixel in noisy_inputs:
            file.write(" " + str(round(pixel, 3)))
          
        file.write("\n")

def run_normal_autoencoder():
  with open("tp5/config/linear_autoencoder.json") as f:
    config = json.load(f)
  ENCODER_LAYERS = config['encoder_layers']

  inputs, labels = read_letters()

  input_len = len(inputs[0])

  layer_sizes = [input_len, *ENCODER_LAYERS, config['latent_space_dim'], *(ENCODER_LAYERS[::-1]), input_len]

  network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1, 1), "autoencoder")
  norm_inputs = network.normalize(inputs.copy(), 0, 1)  
  denormalize = network.gen_denormalize_function(0, 1)

  w = get_weights(config, network, norm_inputs, norm_inputs)

  # network.denormalized_error(norm_inputs, norm_inputs, w, denormalize)
  for i, (x, x2, l) in enumerate(zip(norm_inputs, network.output_function(norm_inputs, w), labels)):
    # print(i,x)
    plot_comparison(denormalize(x), denormalize(x2), f'{l}')
  
  encoder, w_encoder = network.get_encoder(w)
  plot_latent_space(encoder.output_function(norm_inputs, w_encoder), labels, "Latent space")

def run_denoising_autoencoder():
  with open("tp5/config/denoising_autoencoder.json") as f:
    config = json.load(f)
  ENCODER_LAYERS = config['encoder_layers']

  inputs, labels = read_letters()

  generate_noise(labels, inputs)

  input_dict = dict(zip(labels, inputs))
  noisy_inputs, noisy_labels = read_letters("tp5/noisy_font.txt")

  
  input_len = len(inputs[0])
  
  layer_sizes = [input_len, *ENCODER_LAYERS, config['latent_space_dim'], *(ENCODER_LAYERS[::-1]), input_len]

  all_inputs = np.concatenate((inputs, noisy_inputs))
  all_expected = np.concatenate((inputs, np.array([input_dict[l] for l in noisy_labels]))) #[3*i for i in inputs]

  network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), (-1, 1), "autoencoder")
  norm_inputs = network.normalize(all_inputs.copy(), 0, 1)
  norm_expected = network.normalize(all_expected.copy(), 0, 1)
  denormalize = network.gen_denormalize_function(0, 1)

  w = get_weights(config, network, norm_inputs, norm_expected)

  # network.denormalized_error(norm_inputs, norm_inputs, w, denormalize)
  for i, (x, x2, l) in enumerate(zip(norm_inputs, network.output_function(norm_inputs, w), labels)):
    # print(i,x)
    plot_comparison(denormalize(x), denormalize(x2), f'{l}')
  
  # encoder, w_encoder = network.get_encoder(w)
  # plot_latent_space(encoder.output_function(norm_inputs, w_encoder), labels, "Latent space")
  

def get_weights(config: dict, network: MultiLayerNetwork, inputs: np.array, expected_list: np.array) -> np.array:
  if config['import']:
    w = network.import_weights(config['import_path'])
  else:
    t = datetime.now()
    w, _ = network.train_function(config, inputs, expected_list)
    print(datetime.now() - t)
    if w != None:
      network.export_weights(w, config['import_path'])
  return w


if __name__ == "__main__":
  # run_normal_autoencoder()
  run_denoising_autoencoder()