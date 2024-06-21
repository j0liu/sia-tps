from vae import VAENetwork, ErrorType
import numpy as np
import activation_functions as af
import json
from datetime import datetime
from plots import plot_comparison, plot_latent_space, generate_latent_space_grid, plot_output_grid, generate_lerp, plot_all_patterns_together
from PIL import Image
import os
import math

EMOJI_SIZE = (20,20)
def parse_emojis():
    # for each file in emojis dir
    PATH_EMOJIS = "tp5/emojis"
    PATH_TEXTS = "tp5/emojis.txt"
    with open(PATH_TEXTS, 'w') as f:
        for file in os.listdir(PATH_EMOJIS):
            img = Image.open(os.path.join(PATH_EMOJIS, file)).convert('L')  # Convert image to grayscale
            img = img.resize(EMOJI_SIZE)
            img_array = np.array(img)
            label = file.replace(".png","")
            # normalize between -1 and 1
            img_array = img_array / 255 * 2 - 1
            print(img_array)
            plot_comparison(-img_array, img_array, label, height=EMOJI_SIZE[0], width=EMOJI_SIZE[1])
            # Save the image matrix to a file
            f.write(f'{label} ')
            print(img_array)
            f.write(" ".join([str(np.round(i, 3)) for i in img_array.flatten()]))
            f.write("\n")

FILE_NAME = "tp5/emojis.txt"
def read_emojis(file_name = FILE_NAME):
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


def ej_2():
  with open("tp5/config/vae.json") as f:
    config = json.load(f)
  ENCODER_LAYERS = config['encoder_layers']

  inputs, labels = read_emojis()
  # inputs = inputs[:10]
  # labels = labels[:10]
  layer_sizes = VAENetwork.gen_layers(len(inputs[0]), config['latent_space_dim'], ENCODER_LAYERS)

  network = VAENetwork(layer_sizes, af.gen_tanh(config['beta']), af.gen_tanh_derivative(config['beta']), ErrorType.MSE, (-1, 1), "autoencoder")
  
  w = get_weights(config, network, inputs, inputs)
  outputs = network.output_function(inputs, w)
  for i, (x, x2, l) in enumerate(zip(inputs, outputs, labels)):
    plot_comparison(x, x2, f"{config['title']} {l}", height=EMOJI_SIZE[0], width=EMOJI_SIZE[1])

  encoder, w_encoder = network.get_encoder(w)
  latent_space = encoder.output_function(inputs, w_encoder)
  latent_max = math.ceil(max([1] + [np.max(np.abs(l)) for l in latent_space]))
  
  plot_latent_space(latent_space, labels, f"{config['title']}_Latent space", latent_max)


  decoder, w_decoder = network.get_decoder(w)

  output_grid, x_vals, y_vals = generate_latent_space_grid(decoder, w_decoder, (15, 15), latent_max)
  plot_output_grid(-output_grid, x_vals, y_vals, EMOJI_SIZE, title=f"{config['title']}_Output grid")

  output_list, x_vals, y_vals = generate_lerp(decoder, w_decoder, 10, latent_space[0], latent_space[1])
  plot_all_patterns_together(-output_list, zip(x_vals, y_vals), EMOJI_SIZE, title=f"{config['title']}_Lerp")
  



def get_weights(config: dict, network: VAENetwork, inputs: np.array, expected_list: np.array) -> np.array:
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
  ej_2()
#   parse_emojis()