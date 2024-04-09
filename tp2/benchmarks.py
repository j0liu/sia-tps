from maps import CLASS_MAP, CROSSOVER_MAP, MUTATION_MAP, MUTATION_FUNCTION_MAP, REPLACE_MAP, SELECTION_MAP, PAIRING_MAP, STOPPING_MAP
import json
from main import generate_population, iterate
import matplotlib.pyplot as plt
import numpy as np
from player import PlayerClass

def perform_crossover_analysis(config, character_class):

    crossover_methods = list(CROSSOVER_MAP.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness

    for method in crossover_methods:
        iterations_list = []
        fitness_list = []

        for _ in range(50):  # Assuming 50 runs as specified
            # Modify config for current crossover method
            config["crossover"]["method"] = method
            # Generate initial population and run the simulation
            population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))

        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))

    # Update the plot function calls to include standard deviations
    plot_results("Crossover Method", "Avg Iterations", character_class, crossover_methods, avg_iterations_per_method, std_iterations_per_method, False)
    plot_results("Crossover Method", "Avg Fitness", character_class, crossover_methods, avg_fitness_per_method, std_fitness_per_method, True)

def perform_mutation_analysis(config, character_class):

    mutation_methods = list(MUTATION_MAP.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness

    for method in mutation_methods:
        iterations_list = []
        fitness_list = []

        # Perform multiple runs for reliability
        for _ in range(50):  # Assuming 50 runs as specified
            # Modify config for current mutation method
            config["mutation_type"] = method
            # Generate initial population and run the simulation
            population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))

        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))
    
    # Update the plot function calls to include standard deviations
    plot_results("Mutation Method", "Avg Iterations", character_class, mutation_methods, avg_iterations_per_method, std_iterations_per_method, False)
    plot_results("Mutation Method", "Avg Fitness", character_class, mutation_methods, avg_fitness_per_method, std_fitness_per_method, True)

def perform_all_selection_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "roulette": {},
        "universal": {},
        "ranking": {},
        "boltzmann": {'t0': 100, 'tc': 1, 'k': 0.1},
        "deterministic_tournament": {"random_pick_size": 10},
        "probabilistic_tournament": {}
    }
    perform_selection_analysis(config, character_class, method_arguments)


def perform_boltzmann_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "boltzmann 1": {'t0': 100, 'tc': 1, 'k': 10},
        "boltzmann 2": {'t0': 500, 'tc': 50, 'k': 0.01},
        "boltzmann 3": {'t0': 10, 'tc': 1, 'k': 1},
        "boltzmann 4": {'t0': 1000, 'tc': 10, 'k': 1},
    }
    perform_selection_analysis(config, character_class, method_arguments)


def perform_deterministic_tournament_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "deterministic_tournament 1": {"random_pick_size": 2},
        "deterministic_tournament 1": {"random_pick_size": 4},
        "deterministic_tournament 2": {"random_pick_proportion": 0.1},
        "deterministic_tournament 3": {"random_pick_proportion": 0.2},
        "deterministic_tournament 4": {"random_pick_proportion": 0.5},
    }
    perform_selection_analysis(config, character_class, method_arguments)

def perform_selection_analysis(config, character_class, method_arguments):
    selection_methods = list(method_arguments.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    i = 0

    config["selection_coefficient"] = 1
    config["selection2"]["method"] = "elite"
    config["selection2"]["params"] = {} 
    config["selection3"]["method"] = "elite"
    config["selection3"]["params"] = {}
    config["selection4"]["method"] = "elite"
    config["selection4"]["params"] = {}     


    for method_instance in selection_methods:
        # Cambiar la config para cada method
        method = method_instance.split(' ')[0]
        iterations_list = []
        fitness_list = []
        i+=1
        config["selection1"]["method"] = method
        config["selection1"]["params"] = method_arguments[method_instance]
        # Perform multiple runs for reliability
        for _ in range(50):  # Assuming 50 runs as specified
            # Generate initial population and run the simulation
            population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))

        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))
    
    # Update the plot function calls to include standard deviations
    plot_results("Selection Method", "Avg Iterations", character_class, selection_methods, avg_iterations_per_method, std_iterations_per_method, False)
    plot_results("Selection Method", "Avg Fitness", character_class, selection_methods, avg_fitness_per_method, std_fitness_per_method, True)


def perform_replacement_analysis(config, character_class):
    
        replacement_methods = list(REPLACE_MAP.keys())
    
        avg_iterations_per_method = []
        avg_fitness_per_method = []
        std_iterations_per_method = []  # For standard deviation of iterations
        std_fitness_per_method = []     # For standard deviation of fitness
    
        for method in replacement_methods:
            iterations_list = []
            fitness_list = []
    
            # Perform multiple runs for reliability
            for _ in range(50):  # Assuming 50 runs as specified
                # Modify config for current replacement method
                config["replace"] = method
                # Generate initial population and run the simulation
                population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
                populations_list = iterate(population, config)
                # Calculate and store the results of this run
                iterations_list.append(len(populations_list))
                fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
    
            # Calculate averages for this method
            avg_iterations_per_method.append(np.mean(iterations_list))
            avg_fitness_per_method.append(np.mean(fitness_list))
            # Calculate standard deviations for this method
            std_iterations_per_method.append(np.std(iterations_list))
            std_fitness_per_method.append(np.std(fitness_list))
    
        # Update the plot function calls to include standard deviations
        plot_results("Replacement Method", "Avg Iterations", character_class, replacement_methods, avg_iterations_per_method, std_iterations_per_method, False)
        plot_results("Replacement Method", "Avg Fitness", character_class, replacement_methods, avg_fitness_per_method, std_fitness_per_method, True)

def plot_results(x, y, character_class, crossover_methods, data, std_devs, show_digits):
    plt.figure()
    index = np.arange(len(crossover_methods))
    bar_width = 0.35

    # Define a list of colors to cycle through
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Ensure we have enough colors by repeating the list if necessary
    if len(crossover_methods) > len(colors):
        colors = colors * (len(crossover_methods) // len(colors) + 1)

    bars = plt.bar(index, data, bar_width, yerr=std_devs, color=colors[:len(crossover_methods)], capsize=5)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(index, crossover_methods, rotation=45)
    plt.title(f"{x} per {y} for {character_class}")

    # Adding the number on top of each bar
    for bar in bars:
        height = bar.get_height()
        if show_digits:
            plt.text(bar.get_x() + bar.get_width() / 1.5, height, f'{height:.2f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 1.5, height, f'{height:.0f}', ha='center', va='bottom')

    # plt.legend([y])
    plt.tight_layout()
    plt.show()



analysis_map = {
    "crossover" : perform_crossover_analysis,
    "replacement": perform_replacement_analysis,
    "mutation": perform_mutation_analysis,
    "selection": perform_all_selection_analysis,
    "boltzmann": perform_boltzmann_analysis,
    "deterministic_tournament": perform_deterministic_tournament_analysis
}

def main(analysis_names, class_name):
    with open("tp2/config.json") as config_file:
        config = json.load(config_file)

    classes = CLASS_MAP.keys() if class_name == "all" else [class_name]
    for character_class in classes:
        for analysis_name in analysis_names:
            print(f"Analyzing for class: {character_class}")
            config["class"] = character_class  # Set the current class in the config
            analysis_map[analysis_name](config, character_class)        

import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(analysis_map.keys(), "all")
    elif len(sys.argv) == 2:
        main(analysis_map.keys(), sys.argv[1])
    else:
        main(sys.argv[2:], sys.argv[1])
