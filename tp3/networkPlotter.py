import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights):
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    cmap = plt.get_cmap('Blues')  # Get the colormap

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

    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing],
                                  c=cmap(norm(weights[n][m][o])), zorder=1)
                ax.add_artist(line)
                # Adding weight annotations
                if weights:
                    weight = weights[n][m][o]
                    x = n * h_spacing + left + 0.05  # Small horizontal offset from the start node
                    y = layer_top_a - m * v_spacing + (layer_top_b - o * v_spacing - layer_top_a + m * v_spacing) * 0.3
                    ax.text(x, y, f'{weight:.2f}', color='blue', fontsize=8, verticalalignment='center')


def plot_neural_network(layers):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.axis('off')
    weights = [
        [
            [-0.47305217, -0.33284312,  0.15336057,  0.0],
            [-0.91474575, -0.4731593,   0.37041971,  0.0],
            [0.0,         0.02749187,  0.40582895,  0.0],
            [0.0,         0.19039888,  0.88222326,  0.0]
        ],
        [
            [0.0,         0.43801127,  0.5622559,   0.94494164],
            [0.0,         0.8469868,   0.59847222,  0.6251414],
            [0.0,         0.0,         0.0,          0.0],
            [0.0,         0.0,         0.0,          0.0]
        ]
    ]
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, layers, weights)
    plt.show()

plot_neural_network([3, 4, 2])