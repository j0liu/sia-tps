import matplotlib.pyplot as plt
import os

def plot_heatmap(hits, names, x_label, y_label):
    plt.figure(figsize=(10, 8))
    plt.imshow(hits, cmap='magma', interpolation='nearest')
    plt.colorbar(label=y_label)
    plt.title(x_label)
    #change the x and y range
    plt.xticks(range(hits.shape[1]))
    plt.yticks(range(hits.shape[0]))
    # add the names
    for i in range(hits.shape[0]):
        for j in range(hits.shape[1]):
            #make font color white and font size small
            plt.text(j, i, names[i][j], ha='center', va='center', color='black', backgroundcolor='white',fontsize=10)
    plt.savefig(f'tp4/plots/{x_label}.png')
    plt.show()



def plot_unified_distance_matrix(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title('Unified distance matrix')
    plt.show()

def plot_first_principal_component(PC1, names, title):
    plt.figure(figsize=(10, 8))
    plt.bar(names, PC1)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(os.path.join(f'tp4/plots/{title}.png'))
    plt.show()

def plot_energies(energies, title=None):
    # Graficar energía
    plt.figure(figsize=(10, 5))
    plt.plot(energies, marker='o')
    plt.xticks(range(len(energies)))
    plt.title('Energía del sistema a lo largo del tiempo')
    plt.xlabel('Paso')
    plt.ylabel('Energía')
    plt.grid(True)
    if title:
        plt.savefig(f'tp4/plots/hopfield/{title}.png')
    plt.show()


def plot_patterns_over_time(config, patterns_over_time, title=None):
    # Graficar patrones recuperados
    fig, axes = plt.subplots(1, len(patterns_over_time), figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(patterns_over_time[i].reshape(5, 5), cmap='Wistia')
        ax.set_title(f'Paso {i}')
        ax.axis('off')
    if title:
        plt.savefig(f'tp4/plots/hopfield/{title}.png')
    plt.show()