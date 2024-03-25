import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('tp1/metrics/heuristic_results_lvl2.json') as f:
    data = json.load(f)

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

# Compute avg_time and its standard deviation for each condition
for condition in data:
    times = data[condition]['time']
    times_in_seconds = [convert_time_to_seconds(t) for t in times]
    data[condition]['avg_time (s)'] = np.mean(times_in_seconds)
    data[condition]['time_std'] = np.std(times_in_seconds)

# Metrics to plot
metrics = ['visited_nodes', 'border', 'avg_time (s)']

# Plotting setup
bar_width = 0.35
opacity = 0.8
index = np.arange(len(data))

# Create a plot for each metric
for metric in metrics:
    fig, ax = plt.subplots()
    
    values = [data[condition][metric] for condition in data]
    
    if metric == 'avg_time (s)':
        # Include error bars for avg_time
        errors = [data[condition]['time_std'] for condition in data]
        bars = ax.bar(index, values, bar_width, alpha=opacity, color='b', yerr=errors)
    else:
        bars = ax.bar(index, values, bar_width, alpha=opacity, color='b')

    ax.set_xlabel('Conditions')
    ax.set_ylabel(metric.capitalize().replace('_', ' '))
    ax.set_title(f'Comparison of {metric.capitalize().replace("_", " ")} across Heuristics')
    ax.set_xticks(index)
    ax.set_xticklabels(data.keys())
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
