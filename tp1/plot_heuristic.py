import json
import matplotlib.pyplot as plt
import numpy as np

# Provided data
with open('tp1/metrics/heuristic_results.json') as f:
    data = json.load(f)

def convert_time_to_seconds(time_str):
    hours, minutes, seconds = map(float, time_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds

# Compute avg_time for each condition
for condition in data:
    times = data[condition]['time']
    times_in_seconds = [convert_time_to_seconds(t) for t in times]
    data[condition]['avg_time'] = np.mean(times_in_seconds)
    data[condition]['time_std'] = np.std(times_in_seconds)

# Plotting
conditions = list(data.keys())
metrics = ['cost', 'solution_length', 'visited_nodes', 'border', 'avg_time']
bar_width = 0.22
opacity = 0.8

for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    index = np.arange(len(conditions))
    
    for i, condition in enumerate(conditions):
        mean = data[condition][metric] if metric != 'avg_time' else data[condition]['avg_time']
        std = data[        condition][metric] if metric != 'avg_time' else 0  # No std deviation for avg_time
        bars = ax.bar(index + i * bar_width, mean, bar_width, alpha=opacity,
                      color=plt.cm.Set1(i), label=condition)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel('Heuristics')
    ax.set_ylabel(metric.capitalize().replace('_', ' '))
    ax.set_title(f'Comparison of {metric.replace("_", " ").capitalize()} across Conditions')
    ax.set_xticks(index + bar_width / 2 * (len(conditions) - 1))
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

    plt.xticks(rotation=45)  # Rotate condition labels to prevent overlapping
    plt.tight_layout()
    plt.show()