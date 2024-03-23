expanded_data = {
    "NO OPTIMIZATION": {
        "GREEDY": {
            "cost": 118, 
            "solution_length": 119, 
            "visited_nodes": 5663, 
            "border": 1357, 
            "times": ["0:00:01.058909", "0:00:01.074554", "0:00:01.135948", "0:00:01.103778", 
                      "0:00:01.088385", "0:00:01.087761", "0:00:01.106328", "0:00:01.091877", 
                      "0:00:01.099153", "0:00:01.120362"]
        },
        "DFS": {
            "cost": 1314, 
            "solution_length": 1315, 
            "visited_nodes": 17680, 
            "border": 542, 
            "times": ["0:00:01.147028", "0:00:01.154817", "0:00:01.184229", "0:00:01.261553", 
                      "0:00:01.233698", "0:00:01.223703", "0:00:01.240544", "0:00:01.212293", 
                      "0:00:01.217086", "0:00:01.188628"]
        },
        "BFS": {
            "cost": 78, 
            "solution_length": 79, 
            "visited_nodes": 55627, 
            "border": 30, 
            "times": ["0:00:09.323966", "0:00:09.754570", "0:00:15.002703", "0:00:09.501628", 
                      "0:00:09.478428", "0:00:09.537069", "0:00:09.660153", "0:00:14.181353", 
                      "0:00:09.704960", "0:00:09.471233"]
        },
        "A*": {
            "cost": 78, 
            "solution_length": 79, 
            "visited_nodes": 54788, 
            "border": 231, 
            "times": ["0:00:14.636971", "0:00:18.238924", "0:00:15.812331", "0:00:24.980171", 
                      "0:00:15.761529", "0:00:15.920506", "0:00:16.052490", "0:00:19.052635", 
                      "0:00:15.825940", "0:00:15.978757"]
        }
    },
    "OPTIMIZED": {
        "GREEDY": {
            "cost": 124, 
            "solution_length": 125, 
            "visited_nodes": 2219, 
            "border": 638, 
            "times": ["0:00:00.315298", "0:00:00.324564", "0:00:00.296999", "0:00:00.282228", 
                      "0:00:00.280173", "0:00:00.287984", "0:00:00.284147", "0:00:00.283863", 
                      "0:00:00.282707", "0:00:00.297812"]
        },
        "DFS": {
            "cost": 2132, 
            "solution_length": 2133, 
            "visited_nodes": 5460, 
            "border": 843, 
            "times": ["0:00:00.512419", "0:00:00.514325", "0:00:00.521339", "0:00:00.511854", 
                      "0:00:00.519796", "0:00:00.521836", "0:00:00.539880", "0:00:00.538328", 
                      "0:00:00.523069", "0:00:01.238298"]
        },
        "BFS": {
            "cost": 78, 
            "solution_length": 79, 
            "visited_nodes": 26765, 
            "border": 20, 
            "times": ["0:00:02.711432", "0:00:02.705289", "0:00:02.728708", "0:00:02.850390", 
                    "0:00:02.841017", "0:00:03.983269", "0:00:07.608654", "0:00:03.438811", 
                    "0:00:02.727752", "0:00:02.790730"]
        },
        "A*": {
            "cost": 78, 
            "solution_length": 79, 
            "visited_nodes": 26374, 
            "border": 113, 
            "times": ["0:00:03.826317", "0:00:03.922265", "0:00:05.388885", "0:00:06.976151", 
                    "0:00:03.981094", "0:00:03.933396", "0:00:03.940700", "0:00:03.944230", 
                    "0:00:04.004559", "0:00:04.038126"]
        }
    }
}

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
            times = expanded_data[condition][algo]['times']
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