import json
import random
import numpy as np
from functools import partial
from player import (
    Player,
    PlayerClass, 
    PLAYER_GENE_DOMAINS,
    read_population,
    write_population
)
import selection
from maps import CLASS_MAP, CROSSOVER_MAP, MUTATION_MAP, MUTATION_FUNCTION_MAP, REPLACE_MAP, SELECTION_MAP, PAIRING_MAP, STOPPING_MAP


def generate_population(population_size, player_class):
    population = []
    for i in range(population_size):
        h = random.uniform(1.3, 2.0) 
        nums = [random.randint(0, 150) for _ in range(4)] + [150]
        nums.sort()
        stats = nums.copy()
        for i in range(1,5):
            stats[i] = nums[i] - nums[i-1] 
        p = Player(player_class, np.array([h] + stats))
        population.append(p)
    return population

def with_params(map, method_object):
    return partial(map[method_object["method"]], params=method_object["params"])


def iterate(population, config):
    player_class = CLASS_MAP[config["class"]]
    children_count = config["children"]
    stopping_condition = STOPPING_MAP[config["stopping_condition"]]
    pair_genotypes = PAIRING_MAP[config["pairing"]]
    crossover = with_params(CROSSOVER_MAP, config["crossover"])
    mutate = MUTATION_MAP[config["mutation_type"]]
    mutation_rate = config["mutation_rate"]
    mutation_function = MUTATION_FUNCTION_MAP[config["mutation_function"]]
    replace = REPLACE_MAP[config["replace"]]
    selection1 = with_params(SELECTION_MAP, config["selection1"])
    selection2 = with_params(SELECTION_MAP, config["selection2"])
    selection3 = with_params(SELECTION_MAP, config["selection3"])
    selection4 = with_params(SELECTION_MAP, config["selection4"])
    
    pair_select = partial(selection.composite, selection_method1=selection1, selection_method2=selection2, coef_method1=config["selection_coefficient"])
    replacement_select = partial(selection.composite, selection_method1=selection3, selection_method2=selection4, coef_method1=config["replacement_coefficient"])
    
    populations_list = []
    iterations = 0
    while not stopping_condition(config["max_iterations"], iterations, populations_list):
        iterations += 1

        parents = pair_select(population, children_count, iterations)
        children_genotypes = pair_genotypes([p.genotype for p in parents], crossover)
        children = [Player(player_class, mutate(genotype, mutation_function(mutation_rate, generation=iterations),  PLAYER_GENE_DOMAINS)) for genotype in children_genotypes]
        population = replace(population, children, replacement_select, iterations)
        populations_list.append(population)
        
    return populations_list


def main():
    with open("tp2/config.json") as config:
        config = json.load(config)

    POPULATION_SIZE = config["initial_population_size"] 
    PLAYER_CLASS = CLASS_MAP[config["class"]]

    population = generate_population(POPULATION_SIZE, PLAYER_CLASS)
    result = iterate(population, config)
    for g in result:
        print("generation_________________________________________________________________")
        print(max(g, key=lambda p: p.fitness))



if __name__ == "__main__":
    main()