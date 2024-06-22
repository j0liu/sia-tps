import matplotlib.pyplot as plt
import numpy as np
import math

COLOR_MAP = 'gray'

def plot_comparison(original, reconstructed, label, height = 7, width = 5, with_rounded = True):
    plt.figure()
    total_plots = 2 if not with_rounded else 3
    plt.subplot(1, 3, 1)
    plt.imshow(original.reshape(height, width), cmap=COLOR_MAP)
    plt.title(f'Original {label}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed.reshape(height, width), cmap=COLOR_MAP)
    plt.title(f'Reconstructed {label}')
    plt.axis('off')

    if with_rounded:
        plt.subplot(1, 3, 3)
        plt.imshow(np.round(reconstructed.reshape(height, width),0), cmap=COLOR_MAP)
        plt.title(f'Rounded {label}')
        plt.axis('off')

    plt.savefig(f"tp5/plots/{label}.png")
    # plt.show()
    plt.close()


def plot_latent_space(points, labels, title, length=1.2):
    plt.figure()
    unique_labels = np.unique(labels)
    num_unique_labels = len(unique_labels)
    colormap = plt.cm.get_cmap('tab20', num_unique_labels)

    # Create a dictionary to map labels to colors
    label_to_color = {label: colormap(i) for i, label in enumerate(unique_labels)}

    length += 0.2
    plt.xlim(-length, length)
    plt.ylim(-length, length)
    print(points)
    plt.scatter(points[:, 0], points[:, 1])
    for i, p in enumerate(points):
        plt.scatter(p[0], p[1], color=label_to_color[labels[i]], label=labels[i] if labels[i] not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.annotate(labels[i], (p[0], p[1]), color='black')

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))

    plt.title("Latent Space")
    plt.savefig(f"tp5/plots/{title}.png")
    plt.close()

def generate_latent_space_grid(decoder, w_decoder, grid_size=(7, 5), length=1):
    x_vals = np.linspace(-length, length, grid_size[1])
    y_vals = np.linspace(-length, length, grid_size[0])
    latent_space_grid = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1, 2)
    
    outputs = decoder.output_function(latent_space_grid, w_decoder)
    
    return np.array(outputs).reshape(grid_size[0], grid_size[1], -1), x_vals, y_vals


def plot_output_grid(output_grid, x_vals, y_vals, letter_shape=(7, 5), title = ""):
    grid_size = output_grid.shape[:2]
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8),
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            axes[i, j].imshow(output_grid[grid_size[0] - 1 - i, j].reshape(letter_shape), cmap=COLOR_MAP, aspect='auto')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False)
            if j == 0:
                axes[i, j].set_ylabel(f"{y_vals[grid_size[0] - 1 - i]:.2f}", fontsize=8)
            if i == grid_size[0] - 1:
                axes[i, j].set_xlabel(f"{x_vals[j]:.2f}", fontsize=8)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(f"tp5/plots/{title}.png")


def plot_all_comparisons(noisy_inputs, denoised_outputs, labels, noise_level, denormalize, title, letters_per_row=8):
    n = len(noisy_inputs)
    rows = n // letters_per_row + (n % letters_per_row != 0)
    fig, axes = plt.subplots(rows * 2, letters_per_row, figsize=(letters_per_row * 2, rows * 4))
    fig.suptitle(f"Noise Level: {noise_level}")
    
    for i, (noisy, denoised) in enumerate(zip(noisy_inputs, denoised_outputs)):
        row = (i // letters_per_row) * 2
        col = i % letters_per_row
        
        axes[row, col].imshow(denormalize(noisy).reshape(7, 5), cmap='summer')
        axes[row, col].axis('off')
        
        axes[row + 1, col].imshow(np.round(denormalize(denoised).reshape(7, 5)), cmap='summer')
        axes[row + 1, col].axis('off')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"tp5/plots/noise/{title}.png")

def generate_lerp(decoder, w_decoder, length, p1, p2):
    x_vals = np.linspace(p1[0], p2[0], length)
    y_vals = np.linspace(p1[1], p2[1], length)

    lerp_line = np.array(np.column_stack((x_vals, y_vals)))
    
    outputs = decoder.output_function(lerp_line, w_decoder)
    
    return outputs, x_vals, y_vals

def plot_all_patterns_together(patterns, labels, shape, title):
    plt.figure(figsize=(15, 10))

    for idx, (label, pattern) in enumerate(zip(labels,patterns), 1):
        plt.subplot(1, len(patterns), idx)
        plt.imshow(pattern.reshape(shape[0], shape[1]), cmap=COLOR_MAP)
        plt.title(label)
        plt.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"tp5/plots/{title} lerp.png")
