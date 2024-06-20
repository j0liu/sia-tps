import matplotlib.pyplot as plt
import numpy as np
import math
def plot_comparison(original, reconstructed, label):
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

def plot_latent_space(points, labels, title):
    plt.figure()
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    print(points)
    plt.scatter(points[:, 0], points[:, 1])
    for i, p in enumerate(points):
        plt.annotate(labels[i], (p[0], p[1]))

    plt.title(title)
    plt.savefig(f"tp5/plots/{title}.png")
    plt.close()

def generate_latent_space_grid(decoder, w_decoder, grid_size=(7, 5)):
    length = math.pi/2
    x_vals = np.linspace(-length, length, grid_size[1])
    y_vals = np.linspace(-length, length, grid_size[0]) 
    latent_space_grid = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)
    
    outputs = []
    for z in latent_space_grid:
        output = decoder.output_function([z], w_decoder)
        outputs.append(output.flatten())
    
    return np.array(outputs).reshape(grid_size[0], grid_size[1], -1), x_vals, y_vals

def plot_output_grid(output_grid, x_vals, y_vals, letter_shape=(7, 5), title = ""):
    grid_size = output_grid.shape[:2]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8),
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            axes[i, j].imshow(output_grid[i, j].reshape(letter_shape), cmap='gray', aspect='auto')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False) 
            if j == 0:
                axes[i, j].set_ylabel(f"{y_vals[i]:.2f}", fontsize=8)
            if i == grid_size[0] - 1:
                axes[i, j].set_xlabel(f"{x_vals[j]:.2f}", fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.title("Latent space grid")
    plt.savefig(f"tp5/plots/{title}.png")
    