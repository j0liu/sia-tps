import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights):
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


def plot_neural_network(weights, layer_sizes):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, 0.1, 0.9, 0.1, 0.9, layer_sizes, weights)
    plt.show()

if __name__ == "__main__":
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
    plot_neural_network(weights, [4, 4, 4])
