import matplotlib.pyplot as plt
import os
import json

with open("tp3/config/ej2.json") as f:
    config = json.load(f)

def plot_k_fold_errors(errors, train_errors, title):
    if not os.path.exists(f'tp3/plots/EJE2/{config["k"]}'):
        os.makedirs(f'tp3/plots/EJE2/{config["k"]}')

    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors) + 1), errors, label='Test error')
    ax.plot(range(1, len(train_errors) + 1), train_errors, label='Train error')
    ax.set_title(f'Error vs Epoch for {title} | K = {config["k"]}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()
    plt.savefig(f'tp3/plots/EJE2/{config["k"]}/{title}.png')
    plt.close(fig)  # Close the figure to free up memory
