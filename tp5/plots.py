import matplotlib.pyplot as plt
import numpy as np

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
    plt.show()
    plt.close()