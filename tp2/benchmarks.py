from maps import CLASS_MAP, CROSSOVER_MAP, MUTATION_MAP, MUTATION_FUNCTION_MAP, REPLACE_MAP, SELECTION_MAP, PAIRING_MAP, STOPPING_MAP
import json
from main import generate_population, iterate
import matplotlib.pyplot as plt
import numpy as np

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

# def perform_selection_analysis(config, character_class):
    
#         selection_methods = list(SELECTION_MAP.keys())
    
#         avg_iterations_per_method = []
#         avg_fitness_per_method = []
#         std_iterations_per_method = []  # For standard deviation of iterations
#         std_fitness_per_method = []     # For standard deviation of fitness
#         i = 0
#         for method in selection_methods:
#             iterations_list = []
#             fitness_list = []
#             i+=1
    
#             # Perform multiple runs for reliability
#             for _ in range(50):  # Assuming 50 runs as specified
#                 # Modify config for current selection method
#                 config[f"selection{i}"]["method"] = method
#                 # Generate initial population and run the simulation
#                 population = generate_population(config["initial_population_size"], CLASS_MAP[config["class"]])
#                 populations_list = iterate(population, config)
#                 # Calculate and store the results of this run
#                 iterations_list.append(len(populations_list))
#                 fitness_list.append(max(max(p.fitness for p in generation) for generation in populations_list))
    
#             # Calculate averages for this method
#             avg_iterations_per_method.append(np.mean(iterations_list))
#             avg_fitness_per_method.append(np.mean(fitness_list))
#             # Calculate standard deviations for this method
#             std_iterations_per_method.append(np.std(iterations_list))
#             std_fitness_per_method.append(np.std(fitness_list))
        
#         # Update the plot function calls to include standard deviations
#         plot_results("Selection Method", "Avg Iterations", character_class, selection_methods, avg_iterations_per_method, std_iterations_per_method, False)
#         plot_results("Selection Method", "Avg Fitness", character_class, selection_methods, avg_fitness_per_method, std_fitness_per_method, True)

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

    plt.legend([y])
    plt.tight_layout()
    plt.show()


def main():
    with open("tp2/config.json") as config_file:
        config = json.load(config_file)

    for character_class in CLASS_MAP.keys():
        print(f"Analyzing for class: {character_class}")
        config["class"] = character_class  # Set the current class in the config

        # Perform crossover and mutation analysis for the current class
        perform_replacement_analysis(config, character_class)



if __name__ == "__main__":
    main()
