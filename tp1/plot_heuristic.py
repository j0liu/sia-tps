import json
import matplotlib.pyplot as plt
import numpy as np

# Provided data
data = {
    "manhattan": {
        "time": [
            "0:00:00.222662",
            "0:00:00.219559",
            "0:00:00.212130",
            "0:00:00.213832",
            "0:00:00.218907",
            "0:00:00.211350",
            "0:00:00.218085",
            "0:00:00.211479",
            "0:00:00.210780",
            "0:00:00.226201"
        ],
        "cost": 111,
        "solution_length": 112,
        "visited_nodes": 3187,
        "border": 128
    },
    "manhattan+p": {
        "time": [
            "0:00:00.101350",
            "0:00:00.100548",
            "0:00:00.099631",
            "0:00:00.099296",
            "0:00:00.099194",
            "0:00:00.102956",
            "0:00:00.103022",
            "0:00:00.099880",
            "0:00:00.099577",
            "0:00:00.100827"
        ],
        "cost": 116,
        "solution_length": 117,
        "visited_nodes": 1033,
        "border": 430
    },
    "mod_manhattan+p": {
        "time": [
            "0:00:01.119715",
            "0:00:01.114321",
            "0:00:01.122529",
            "0:00:01.124060",
            "0:00:01.129984",
            "0:00:01.144459",
            "0:00:01.142064",
            "0:00:01.126369",
            "0:00:01.136909",
            "0:00:01.127226"
        ],
        "cost": 130,
        "solution_length": 131,
        "visited_nodes": 814,
        "border": 572
    },
    "max_manhattan+p": {
        "time": [
            "0:00:01.194612",
            "0:00:01.185617",
            "0:00:01.186292",
            "0:00:01.181885",
            "0:00:01.177766",
            "0:00:01.187881",
            "0:00:01.181640",
            "0:00:01.192413",
            "0:00:01.191343",
            "0:00:01.175399"
        ],
        "cost": 130,
        "solution_length": 131,
        "visited_nodes": 814,
        "border": 572
    }
}

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