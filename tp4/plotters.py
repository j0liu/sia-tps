import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

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
    # Sort the data
    sorted_indices = np.argsort(PC1)[::-1]
    sorted_PC1 = PC1[sorted_indices]
    sorted_names = np.array(names)[sorted_indices]
    
    plt.figure(figsize=(10, 8))
    plt.bar(sorted_names, sorted_PC1)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(os.path.join(f'./tp4/plots/{title}.png'))
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
        plt.savefig(f'./tp4/plots/hopfield/{title}.png')
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

custom_palette = [
    "#FF0000", "#FF6347",  # Red, Light Coral
    "#FF4500", "#FFA07A",  # OrangeRed, Light Salmon
    "#FFA500", "#FFDAB9",  # Orange, PeachPuff
    "#FFFF00", "#FFD700",  # Yellow, Light Goldenrod Yellow (adjusted)
    "#7CFC00", "#ADFF2F",  # LawnGreen, GreenYellow (adjusted)
    "#00FF00", "#7FFF00",  # Lime, Chartreuse
    "#20B2AA", "#7FFFD4",  # LightSeaGreen, Aquamarine
    "#00FFFF", "#AFEEEE",  # Aqua, PaleTurquoise
    "#0000FF", "#87CEEB",  # Blue, SkyBlue (adjusted)
    "#8A2BE2", "#9370DB",  # BlueViolet, MediumPurple (adjusted)
    "#FF00FF", "#DA70D6",  # Magenta, Orchid
    "#FF1493", "#FFB6C1",  # DeepPink, LightPink
    "#8B4513", "#F5DEB3",  # SaddleBrown, Wheat
    "#000000", "#D3D3D3"   # Black, LightGrey
]


def plot_biplot(pca_data, loadings, names, variable_names):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=names, palette='tab10', legend=None)

    for i in range(pca_data.shape[0]):
        plt.text(pca_data[i, 0], pca_data[i, 1], names[i], ha='center', va='bottom', fontsize=8, color='black')

    for i, (loading, name) in enumerate(zip(loadings.T, variable_names)):
        plt.arrow(0, 0, loading[0], loading[1], color='r', alpha=0.5)
        plt.text(loading[0]*1.15, loading[1]*1.15, name, color='g', ha='center', va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Biplot")
    plt.grid()
    plt.show()
    
def plot_boxplots(nostd_data, standarized_data, variable_names):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # Non-standardized data
    sns.boxplot(data=nostd_data, ax=axes[0])
    axes[0].set_xticklabels(variable_names, rotation=90)
    axes[0].set_title('Non-Standardized Data')

    # Standardized data
    sns.boxplot(data=standarized_data, ax=axes[1])
    axes[1].set_xticklabels(variable_names, rotation=90)
    axes[1].set_title('Standardized Data')

    plt.tight_layout()
    plt.show()
