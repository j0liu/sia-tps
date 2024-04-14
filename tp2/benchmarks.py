from maps import CLASS_MAP, CROSSOVER_MAP, MUTATION_MAP, MUTATION_FUNCTION_MAP, REPLACE_MAP, SELECTION_MAP, PAIRING_MAP, STOPPING_MAP, BOLTZMANN_MAP, DETERMINISTIC_MAP
import json
from main import generate_population, iterate
import matplotlib.pyplot as plt
import numpy as np
from player import PlayerClass

ITERATIONS = 50

def perform_crossover_analysis(config, character_class):

    methods = list(CROSSOVER_MAP.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []

    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for method in methods:
        print(method)
        genealogies = []
        iterations_list = []
        fitness_list = []
        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Modify config for current crossover method
            config["crossover"]["method"] = method
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))

    # Update the plot function calls to include standard deviations
    plot_general_results("Crossover", character_class, methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, methods)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method

def perform_mutation_analysis(config, character_class):

    mutation_methods = list(MUTATION_MAP.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []
    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for method in mutation_methods:
        print(method)
        genealogies = []
        iterations_list = []
        fitness_list = []

        # Perform multiple runs for reliability
        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Modify config for current mutation method
            config["mutation_type"] = method
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        print("min ", min(iterations_list),"max ", max(iterations_list),"avg ", np.mean(iterations_list),"std ", np.std(iterations_list))
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))
    
    # Update the plot function calls to include standard deviations
    plot_general_results("Mutation", character_class, mutation_methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, mutation_methods)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method

def perform_all_crossover_selection_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "roulette": {},
        "universal": {},
        "ranking": {},
        "boltzmann": {'t0': 100, 'tc': 1, 'k': 0.1},
        "deterministic_tournament": {"random_pick_size": 10},
        "probabilistic_tournament": {}
    }
    return perform_crossover_selection_analysis(config, character_class, method_arguments)

def perform_crossover_selection_analysis(config, character_class, method_arguments):
    selection_methods = list(method_arguments.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []

    config["selection_coefficient"] = 0.5
    config["replacement_coefficient"] = 1
    config["selection3"]["method"] = "elite"
    config["selection3"]["params"] = {}
    config["selection4"]["method"] = "elite"
    config["selection4"]["params"] = {}

    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for method_instance in selection_methods:
        # Cambiar la config para cada method
        print(method_instance)
        method = method_instance.split(' ')[0]
        genealogies = []
        iterations_list = []
        fitness_list = []
        config["selection1"]["method"] = method
        config["selection1"]["params"] = method_arguments[method_instance]
        config["selection2"]["method"] = method
        config["selection2"]["params"] = method_arguments[method_instance]
        # Perform multiple runs for reliability
        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))
    
    # Update the plot function calls to include standard deviations
    plot_general_results("Selection", character_class, selection_methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, selection_methods)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method

def perform_all_replacement_selection_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "roulette": {},
        "universal": {},
        "ranking": {},
        "boltzmann": {'t0': 100, 'tc': 1, 'k': 0.1},
        "deterministic_tournament": {"random_pick_size": 10},
        "probabilistic_tournament": {}
    }
    return perform_replacement_selection_analysis(config, character_class, method_arguments)

def perform_boltzmann_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "boltzmann 1": {'t0': 100, 'tc': 1, 'k': 0.1},
        "boltzmann 2": {'t0': 500, 'tc': 50, 'k': 0.01},
        "boltzmann 3": {'t0': 10, 'tc': 1, 'k': 1},
        "boltzmann 4": {'t0': 1000, 'tc': 10, 'k': 1},
        "boltzmann 5": {'t0': 1000, 'tc': 100, 'k': 1},
        "boltzmann 6": {'t0': 1000, 'tc': 500, 'k': 1},
    }
    return perform_replacement_selection_analysis(config, character_class, method_arguments)


def perform_deterministic_tournament_analysis(config, character_class):
    method_arguments = {
        "elite": {},
        "deterministic_tournament 1": {"random_pick_size": 2},
        "deterministic_tournament 2": {"random_pick_size": 4},
        "deterministic_tournament 3": {"random_pick_proportion": 0.1},
        "deterministic_tournament 4": {"random_pick_proportion": 0.2},
        "deterministic_tournament 5": {"random_pick_proportion": 0.5},
        "deterministic_tournament 7": {"random_pick_proportion": 0.9},
    }
    return perform_replacement_selection_analysis(config, character_class, method_arguments)


def perform_replacement_selection_analysis(config, character_class, method_arguments):
    selection_methods = list(method_arguments.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []

    config["selection_coefficient"] = 0.5
    config["selection1"]["method"] = "elite"
    config["selection1"]["params"] = {}
    config["selection2"]["method"] = "deterministic_tournament"
    config["selection2"]["params"] = { "random_pick_proportion": 0.4 }
    config["replacement_coefficient"] = 0.5
    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for method_instance in selection_methods:
        # Cambiar la config para cada method
        print(method_instance)
        method = method_instance.split(' ')[0]
        genealogies = []
        iterations_list = []
        fitness_list = []
        config["selection3"]["method"] = method
        config["selection3"]["params"] = method_arguments[method_instance]
        config["selection4"]["method"] = method
        config["selection4"]["params"] = method_arguments[method_instance]
        # Perform multiple runs for reliability
        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))
    
    # Update the plot function calls to include standard deviations
    plot_general_results("Selection", character_class, selection_methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, selection_methods)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method


def perform_replacement_analysis(config, character_class):
    
        replacement_methods = list(REPLACE_MAP.keys())
    
        avg_iterations_per_method = []
        avg_fitness_per_method = []
        std_iterations_per_method = []  # For standard deviation of iterations
        std_fitness_per_method = []     # For standard deviation of fitness
        genealogies_per_method = []
        population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
    
        for method in replacement_methods:
            print(method)
            genealogies = []
            iterations_list = []
            fitness_list = []
    
            # Perform multiple runs for reliability
            for _ in range(ITERATIONS):  # Assuming 50 runs as specified
                # Modify config for current replacement method
                config["replace"] = method
                # Generate initial population and run the simulation
                populations_list = iterate(population, config)
                # Calculate and store the results of this run
                iterations_list.append(len(populations_list))
                fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
                genealogies.append([max(p.fitness for p in g) for g in populations_list])
            
            genealogies_per_method.append(genealogies)
            # Calculate averages for this method
            avg_iterations_per_method.append(np.mean(iterations_list))
            avg_fitness_per_method.append(np.mean(fitness_list))
            # Calculate standard deviations for this method
            std_iterations_per_method.append(np.std(iterations_list))
            std_fitness_per_method.append(np.std(fitness_list))
    
        # Update the plot function calls to include standard deviations
        plot_general_results("Replacement", character_class, replacement_methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
        plot_evolution(character_class, genealogies_per_method, replacement_methods)
        return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method

def perform_mutation_func_analysis(config, character_class):
    functions = list(MUTATION_FUNCTION_MAP.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []
    
    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for fun in functions:
        print(fun)
        genealogies = []
        iterations_list = []
        fitness_list = []

        # Perform multiple runs for reliability
        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Modify config for current replacement method
            config["mutation_function"] = fun
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))

    # Update the plot function calls to include standard deviations
    plot_general_results("Mutation function", character_class, functions, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, functions)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method


stopping_condition_map = {
    "max_generations": {"max_generations": 100},
    "structure": {"streak_length": 20, "similarity": 0.6},
    "content": {"streak_length": 20},
    # "around_optimal": {"threshold": 100}
}

def perform_stopping_analysis(config, character_class):
    functions = list(stopping_condition_map.keys())

    avg_iterations_per_method = []
    avg_fitness_per_method = []
    std_iterations_per_method = []  # For standard deviation of iterations
    std_fitness_per_method = []     # For standard deviation of fitness
    genealogies_per_method = []
    
    population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])

    for cond in stopping_condition_map:
        print(cond)
        genealogies = []
        iterations_list = []
        fitness_list = []

        for _ in range(ITERATIONS):  # Assuming 50 runs as specified
            # Modify config for current replacement method
            config["stopping_condition"]["method"] = cond
            config["stopping_condition"]["params"] = stopping_condition_map[cond]
            # Generate initial population and run the simulation
            populations_list = iterate(population, config)
            # Calculate and store the results of this run
            iterations_list.append(len(populations_list))
            fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
            genealogies.append([max(p.fitness for p in g) for g in populations_list])
        genealogies_per_method.append(genealogies)
        # Calculate averages for this method
        avg_iterations_per_method.append(np.mean(iterations_list))
        avg_fitness_per_method.append(np.mean(fitness_list))
        # Calculate standard deviations for this method
        std_iterations_per_method.append(np.std(iterations_list))
        std_fitness_per_method.append(np.std(fitness_list))

    plot_general_results("Stopping condition", character_class, functions, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method)
    plot_evolution(character_class, genealogies_per_method, functions)
    return avg_iterations_per_method, avg_fitness_per_method, std_iterations_per_method, std_fitness_per_method

        
def plot_results(x, y, character_class, methods, data, std_devs, show_digits):
    plt.figure()
    index = np.arange(len(methods))
    bar_width = 0.35

    # Define a list of colors to cycle through
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Ensure we have enough colors by repeating the list if necessary
    if len(methods) > len(colors):
        colors = colors * (len(methods) // len(colors) + 1)

    bars = plt.bar(index, data, bar_width, yerr=std_devs, color=colors[:len(methods)], capsize=5)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(index, methods, rotation=45)
    plt.title(f"{x} per {y} for {character_class}")

    # Adding the number on top of each bar
    for bar in bars:
        height = bar.get_height()
        if show_digits:
            plt.text(bar.get_x() + bar.get_width() / 1, height, f'{height:.2f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 1, height, f'{height:.0f}', ha='center', va='bottom')

    # plt.legend([y])
    plt.tight_layout()
    plt.show()

def plot_general_results(aspect, character_class, methods, avg_iterations_per_method, std_iterations_per_method, avg_fitness_per_method, std_fitness_per_method):
    figure, axis = plt.subplots(1,2, figsize=(16, 8))
    subplot_results(figure, axis[0], f"{aspect} Method", "Avg Iterations", character_class, methods, avg_iterations_per_method, std_iterations_per_method, False)
    subplot_results(figure, axis[1], f"{aspect} Method", "Avg Fitness", character_class, methods, avg_fitness_per_method, std_fitness_per_method, True)
    plt.show()


def subplot_results(figure, axis, x, y, character_class, methods, data, std_devs, show_digits):
    #plt.figure()
    index = np.arange(len(methods))
    bar_width = 0.35

    # Define a list of colors to cycle through
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # Ensure we have enough colors by repeating the list if necessary
    if len(methods) > len(colors):
        colors = colors * (len(methods) // len(colors) + 1)
    std_devs = [min(data[i], std_devs[i]) for i in range(len(std_devs))]
    bars = axis.bar(index, data, bar_width, yerr=std_devs, color=colors[:len(methods)], capsize=5)

    plt.sca(axis)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(index, methods, rotation=45)
    plt.title(f"{x} per {y} for {character_class}")

    # Adding the number on top of each bar
    for bar in bars:
        height = bar.get_height()
        if show_digits:
            plt.text(bar.get_x() + bar.get_width() / 1, height, f'{height:.2f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 1, height, f'{height:.0f}', ha='center', va='bottom')

    # plt.legend([y])
    plt.tight_layout()


def plot_evolution(character_class, genealogies_per_method, methods):
    fig, axis = plt.subplots(1, len(genealogies_per_method), figsize=(20, 5))
    for i, genealogies in enumerate(genealogies_per_method):
        for genealogy in genealogies:
            axis[i].plot(range(len(genealogy)), genealogy)
        axis[i].set_title(methods[i])
        axis[i].set_xlabel("Generations")
        axis[i].set_ylabel("Max Fitness")
    plt.suptitle(f"Evolution of Max Fitness for {character_class}")
    plt.tight_layout()
    plt.show()


analysis_map = {
    "crossover" : perform_crossover_analysis,
    "replacement": perform_replacement_analysis,
    "mutation": perform_mutation_analysis,
    "crossover_selection": perform_all_crossover_selection_analysis,
    "boltzmann": perform_boltzmann_analysis,
    "deterministic_tournament": perform_deterministic_tournament_analysis,
    "replacement_selection": perform_all_replacement_selection_analysis,
    "mutation_function": perform_mutation_func_analysis,
    "stopping_condition": perform_stopping_analysis
}

analysis_to_map = {
    "crossover": CROSSOVER_MAP,
    "mutation": MUTATION_MAP,
    "replacement": REPLACE_MAP,
    "crossover_selection": SELECTION_MAP,  # Assuming SELECTION_MAP exists and is relevant
    "boltzmann": BOLTZMANN_MAP,  # Example, adjust based on actual use
    "deterministic_tournament": DETERMINISTIC_MAP,  # Example, adjust based on actual use
    "replacement_selection": SELECTION_MAP,
    "mutation_function": MUTATION_FUNCTION_MAP,
    "stopping_condition": stopping_condition_map
}

def main(analysis_names, class_name):
    with open("tp2/config.json") as config_file:
        config = json.load(config_file)

    classes = CLASS_MAP.keys() if class_name == "all" else [class_name]
    all_results = {name: [] for name in analysis_names}  # Initialize a dict to hold results for each analysis type

    for character_class in classes:
        for analysis_name in analysis_names:
            print(f"Analyzing for class: {character_class}")
            config["class"] = character_class  # Set the current class in the config
            results = analysis_map[analysis_name](config, character_class)
            all_results[analysis_name].append(results)        

    # Compute averages across all character classes for each analysis type
    if class_name != "all": return

    for analysis_name, results in all_results.items():
        method_keys = list(analysis_to_map[analysis_name].keys()) if analysis_name in analysis_to_map else []
        avg_iterations = np.mean([res[0] for res in results], axis=0)
        avg_fitness = np.mean([res[1] for res in results], axis=0)
        std_iterations = np.mean([res[2] for res in results], axis=0)
        std_fitness = np.mean([res[3] for res in results], axis=0)

        # Now you can plot the averaged results
        plot_general_results(analysis_name, "All Characters", method_keys, avg_iterations, std_iterations, avg_fitness, std_fitness)

import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main(analysis_map.keys(), "all")
    elif len(sys.argv) == 2:
        main(analysis_map.keys(), sys.argv[1])
    else:
        main(sys.argv[2:], sys.argv[1])
