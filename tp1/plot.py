import json

with open('tp1/metrics/level2.json', 'r') as f:
    data = json.load(f)

expanded_data = data 

import matplotlib.pyplot as plt
import numpy as np

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

def get_time_stats(times):
    """Convert list of time strings to average seconds and standard deviation."""
    seconds = [convert_time_to_seconds(time) for time in times]
    avg_seconds = np.mean(seconds)
    std_seconds = np.std(seconds)
    return avg_seconds, std_seconds

def plot_grouped_bars_with_annotations(expanded_data):
    for condition in expanded_data:
        for algo in expanded_data[condition]:
            times = expanded_data[condition][algo]['time']
            times_in_seconds = [convert_time_to_seconds(t) for t in times]
            expanded_data[condition][algo]['avg_time'] = np.mean(times_in_seconds)
            expanded_data[condition][algo]['time_std'] = np.std(times_in_seconds)

    algorithms = list(expanded_data['NO OPTIMIZATION'].keys())
    metrics = ['cost', 'solution_length', 'visited_nodes', 'border', 'avg_time']
    conditions = list(expanded_data.keys())

    bar_width = 0.15
    opacity = 0.8

    for metric_idx, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(12, 7))
        index = np.arange(len(algorithms))
        
        for condition_idx, condition in enumerate(conditions):
            means = [expanded_data[condition][algo][metric] for algo in algorithms]
            stds = [expanded_data[condition][algo].get('time_std', 0) if metric == 'avg_time' else 0 for algo in algorithms]
            bars = ax.bar(index + condition_idx * bar_width, means, bar_width, alpha=opacity,
                          color=plt.cm.Set1(condition_idx), label=f'{condition}',
                          yerr=stds, capsize=5)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom')

        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric.capitalize().replace('_', ' '))
        ax.set_title(f'Comparison of {metric.capitalize().replace("_", " ")} across Algorithms and Conditions')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(algorithms)
        ax.legend()

        plt.tight_layout()
        plt.show()

plot_grouped_bars_with_annotations(expanded_data)