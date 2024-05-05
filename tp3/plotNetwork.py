import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
from PIL import Image
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights = None, values = None):
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    cmap = plt.get_cmap('brg')  # Get the colormap

    # Normalize the weight values to 0-1 for the color mapping
    if weights is not None:
        all_weights = [w for sublist in weights for subsublist in sublist for w in subsublist]
        norm = mcolors.Normalize(vmin=min(all_weights), vmax=max(all_weights))

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if values is not None:
                ax.text(n * h_spacing + left, layer_top - m * v_spacing, f'{values[n][m]:.3f}', fontsize=14, verticalalignment='center', horizontalalignment='center', zorder = 5)

    # Edges
    for n, (layer_size_prev, layer_size_next) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_prev - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_next - 1) / 2. + (top + bottom) / 2.
        for current in range(layer_size_next):
            for prev in range(layer_size_prev):
                if weights is None or abs(weights[n][current][prev]) > 0:
                    #random color
                    cm = cmap(np.random.rand())
                    width = 5 * (min(1, max(abs(weights[n][current][prev]),0.2)) if weights is not None else 1)
                    #cm = cmap(abs(weights[n][current][prev])) if weights is not None else 'black'
                    #cm = cmap(norm(abs(weights[n][current][prev])))
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                    [layer_top_a - prev * v_spacing, layer_top_b - current * v_spacing],
                                    c=cm, zorder=1, linewidth = width)
                    ax.add_artist(line)
                    # Adding weight annotations
                    if weights is not None:
                        weight = weights[n][current][prev]
                        x = n * h_spacing + left + 0.05  # Small horizontal offset from the start node
                        y = layer_top_a - prev * v_spacing + (layer_top_b - current * v_spacing - layer_top_a + prev * v_spacing) * 0.3
                        ax.text(x, y, f'{weight:.5f}', color=cm, bbox=dict(facecolor='white', alpha=0.5, edgecolor=cm), fontsize=8, verticalalignment='center', zorder = 5)


def plot_neural_network(layer_sizes, weights=None, values=None):
    if not os.path.exists('tp3/plots/eje3/networks'):
        os.makedirs('tp3/plots/eje3/networks')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, layer_sizes, weights, values)
    # plt.show()
    plt.savefig(f'tp3/plots/eje3/networks/network.png')


def create_network_gif(network, weight_history, input, name):
    gif_images = []
    if not os.path.exists('tp3/plots/eje3/networks/gif_frames'):
        os.makedirs('tp3/plots/eje3/networks/gif_frames')

    # for i, weights in enumerate(weight_history[::10]):
    for i, weights in enumerate(weight_history):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.gca()
        ax.axis('off')

        values = network._forward_propagation(input, weights)
        
        draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, network.layer_sizes, weights, values)
        plt.savefig(f'tp3/plots/eje3/networks/gif_frames/frame_{i}.png')
        plt.close()
        
        # Open the saved image with PIL and append it to the list
        img = Image.open(f'tp3/plots/eje3/networks/gif_frames/frame_{i}.png')
        gif_images.append(img)
    
    # Save the images as a GIF
    gif_images[0].save(f'tp3/plots/eje3/networks/{name}.gif',
                       save_all=True, append_images=gif_images[1:], optimize=False, duration=500, loop=0)
    
# def make_gif_from_folder(folder_path, gif_name):
#     gif_images = []
#     for i, file in enumerate(sorted(os.listdir(folder_path))):
#         img = Image.open(f'{folder_path}/{file}')
#         gif_images.append(img)
#     gif_images[0].save(f'{folder_path}/{gif_name}.gif',
#                        save_all=True, append_images=gif_images[1:], optimize=False, duration=100, loop=0)
    
# if __name__ == "__main__":
#     make_gif_from_folder('tp3/plots/eje3/networks/gif_frames', 'xor')