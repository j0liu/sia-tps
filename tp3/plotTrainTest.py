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
    ax.set_ybound(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()
    plt.savefig(f'tp3/plots/EJE2/{config["k"]}/{title}.png')
    plt.close(fig)  # Close the figure to free up memory


#metrics_to_plot.append([accuracy(test_confusion_matrix), precision(test_confusion_matrix), recall(test_confusion_matrix), f1_score(test_confusion_matrix), accuracy(train_confusion_matrix), precision(train_confusion_matrix), recall(train_confusion_matrix), f1_score(train_confusion_matrix)])
# plot 1 graph for each metric vs epoch
def plot_metrics(metrics, title):
    # Create the directory if it does not exist
    directory = f'tp3/plots/eje3'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Metrics names
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Loop over each metric
    for i, metric_name in enumerate(metric_names):
        fig, ax = plt.subplots()
        # Plot training data for the metric
        ax.plot(range(1, len(metrics) + 1), [m[i + 4] for m in metrics], label=f'Train {metric_name}', color='b')
        # Plot testing data for the metric
        ax.plot(range(1, len(metrics) + 1), [m[i] for m in metrics], label=f'Test {metric_name}', color='r')
        ax.set_title(f'{metric_name} vs Epoch for {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)

        #set ticks every 5
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.legend()
        plt.savefig(f'{directory}/{title}_{metric_name.lower()}_metrics.png')
        plt.close(fig)  # Close the figure to free up memory