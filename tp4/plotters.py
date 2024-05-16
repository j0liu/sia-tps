import matplotlib.pyplot as plt

def plot_heatmap(hits, names):
    plt.figure(figsize=(10, 8))
    plt.imshow(hits, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Entries amount')
    plt.title('Final entries per neuron')
    #change the x and y range
    plt.xticks(range(hits.shape[1]))
    plt.yticks(range(hits.shape[0]))
    # add the names
    for i in range(hits.shape[0]):
        for j in range(hits.shape[1]):
            #make font color white and font size small
            plt.text(j, i, names[i][j], ha='center', va='center', color='green', fontsize=5.5)
    plt.show()

def plot_unified_distance_matrix(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Distance')
    plt.title('Unified distance matrix')
    plt.show()