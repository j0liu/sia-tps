import matplotlib.pyplot as plt
import os

def plot_k_fold_errors(errors, train_errors, title):

    if(not os.path.exists(f'tp3/plots/EJE2')):
        os.makedirs(f'tp3/plots/EJE2')

    fig, ax = plt.subplots()
    ax.plot(errors, label='Test Error')
    ax.plot(train_errors, label='Train Error')
    ax.set_title(f'Error vs Fold for {title}')
    ax.set_xlabel('Fold')
    ax.set_xticks(range(len(errors)))
    ax.set_ylabel('Error')
    ax.legend()
    plt.savefig(f'tp3/plots/EJE2/{title}.png')
    plt.close(fig)  # Close the figure to free up memory