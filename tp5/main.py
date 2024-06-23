# read font.h and parse the letters

from multilayer import MultiLayerNetwork, hypercube_layers, ErrorType
import numpy as np
import activation_functions as af
import json
from datetime import datetime
from plots import plot_comparison, plot_latent_space, generate_latent_space_grid, plot_output_grid, plot_all_comparisons, set_color_map


FILE_NAME = "tp5/font.txt"

def read_letters(file_name = FILE_NAME):
  with open(file_name, "r") as file:
    lines = file.readlines()
  inputs = []
  labels = []
  for line in lines:
    if not line.startswith("//"):
      x = np.array([float(i) for i in line.replace('\n', '').split(' ')[1:]])
      inputs.append(x)
      labels.append(line.split(' ')[0])
  inputs = np.array(inputs)#[:-1]
  labels = labels[:len(inputs)]
  return inputs, labels

def generate_noise(labels: str, inputs : np.array, noise_level: float = 0.1):
  with open(f"tp5/noisy_font.txt", 'w') as file:
    for i, num_pixels in enumerate(inputs):
      noise = np.random.normal(0, noise_level, num_pixels.shape)
      noisy_inputs = np.clip(num_pixels + noise, 0, 1)

      file.write(labels[i])
      for pixel in noisy_inputs:
          file.write(" " + str(round(pixel, 3)))
        
      file.write("\n")

def ej_1a():
  with open("tp5/config/linear_autoencoder.json") as f:
    config = json.load(f)
  ENCODER_LAYERS = config['encoder_layers']

  inputs, labels = read_letters()
  # inputs = inputs[:10]
  # labels = labels[:10]
  layer_sizes = MultiLayerNetwork.gen_layers(len(inputs[0]), config['latent_space_dim'], ENCODER_LAYERS)

  network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), ErrorType.MSE, (-1, 1), "autoencoder")
  norm_inputs = network.normalize(inputs.copy(), 0, 1)  
  denormalize = network.gen_denormalize_function(0, 1)

  set_color_map('summer')
  w = get_weights(config, network, norm_inputs, norm_inputs)
  for i, (x, l) in enumerate(zip(norm_inputs, labels)):
    o = network.forward_propagation(x, w)
    plot_comparison(denormalize(-x), denormalize(-o), f"{config['title']} {l}")
    o_neg = network.forward_propagation(-x, w)
    plot_comparison(denormalize(x), denormalize(-o_neg), f"{config['title']} {l} neg")
    
  encoder, w_encoder = network.get_encoder(w)
  latent_space = encoder.output_function(norm_inputs, w_encoder)
  plot_latent_space(latent_space, labels, f"{config['title']}_Latent space")

  decoder, w_decoder = network.get_decoder(w)
  set_color_map('binary')
  output_grid, x_vals, y_vals = generate_latent_space_grid(decoder, w_decoder, grid_size=(config.get('grid_size', 15), config.get('grid_size', 15)))
  plot_output_grid(output_grid, x_vals, y_vals, letter_shape=(7, 5), title=f"{config['title']}_Output grid")
  

def ej_1b():
    with open("tp5/config/denoising_autoencoder.json") as f:
        config = json.load(f)
    ENCODER_LAYERS = config['encoder_layers']

    inputs, labels = read_letters('tp5/font.txt')

    input_dict = dict(zip(labels, inputs))
    noisy_inputs, noisy_labels = read_letters("tp5/complete_noise.txt")
  
    layer_sizes = MultiLayerNetwork.gen_layers(len(inputs[0]), config['latent_space_dim'], ENCODER_LAYERS)

    all_inputs = np.concatenate((inputs, noisy_inputs))
    all_expected = np.concatenate((inputs, np.array([input_dict[l] for l in noisy_labels]))) #[3*i for i in inputs]
    all_labels = labels + noisy_labels

    network = MultiLayerNetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), ErrorType.MSE, (-1, 1), "autoencoder")
    norm_inputs = network.normalize(all_inputs.copy(), 0, 1)
    norm_expected = network.normalize(all_expected.copy(), 0, 1)
    denormalize = network.gen_denormalize_function(0, 1)

    w = get_weights(config, network, norm_inputs, norm_expected)
    # w = network.import_weights(f"tp5/weights/{config['title']}.txt")

    network.denormalized_error(norm_inputs, norm_inputs, w, denormalize)
    for i, (x, x2, l) in enumerate(zip(norm_inputs, network.output_function(norm_inputs, w), labels)):
      plot_comparison(denormalize(x), denormalize(x2), f"{config['title']} {l}")

    noise_levels = [0.15, 0.3, 0.5, 0.8]  # Adjust noise levels as needed
    for noise_level in noise_levels:
        generate_noise(labels, inputs, noise_level=noise_level)
        noisy_inputs, _ = read_letters("tp5/noisy_font.txt")
        
        norm_noisy_inputs = network.normalize(noisy_inputs.copy(), 0, 1)
        denoised_outputs = network.output_function(norm_noisy_inputs, w)

        plot_all_comparisons(norm_noisy_inputs, denoised_outputs, labels, noise_level, denormalize, f"{config['title']}_{noise_level}")

    encoder, w_encoder = network.get_encoder(w)
    plot_latent_space(encoder.output_function(norm_inputs, w_encoder), all_labels, f"{config['title']}_Latent space")
    decoder, w_decoder = network.get_decoder(w)

    output_grid, x_vals, y_vals = generate_latent_space_grid(decoder, w_decoder, grid_size=(15, 15))
    plot_output_grid(output_grid, x_vals, y_vals, letter_shape=(7, 5), title=f'{config["title"]}_Output grid')

  

def get_weights(config: dict, network: MultiLayerNetwork, inputs: np.array, expected_list: np.array) -> np.array:
  if config['import']:
    w = network.import_weights(f"tp5/weights/{config['title']}.txt")
  else:
    t = datetime.now()
    w, _ = network.train_function(config, inputs, expected_list)
    print(datetime.now() - t)
    network.export_weights(w, f"tp5/weights/{config['title']}.txt")
    print(config)
  return w


if __name__ == "__main__":
  # ej_1a()
  # inputs, labels = read_letters()
  # for noise_lvl in [0,15, 0.3, 0.35, 0.4, 0.45, 0.5]:
  #   generate_noise(labels, inputs, noise_lvl)
  ej_1b()
