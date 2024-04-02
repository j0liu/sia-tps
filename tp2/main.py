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
import mutation
import replacement
import crossover
import pairing
import mutation_func

CLASS_MAP = {
    "warrior": PlayerClass.WARRIOR,
    "archer": PlayerClass.ARCHER,
    "defender": PlayerClass.DEFENDER,
    "infiltrate": PlayerClass.INFILTRATE
}

CROSSOVER_MAP = {
    "onepoint": crossover.one_point,
    "twopoint": crossover.two_point,
    "anular": crossover.anular,
    "uniform": crossover.uniform,
}

MUTATION_MAP = {
  "gene": mutation.mutate_gene,
  "multigene": mutation.mutate_multigene,
  "multigene_limited": mutation.mutate_multigene_limited,
  "none": mutation.mutate_none
}

SELECTION_MAP = {
    "elite" : selection.elite,
    "roulette" : selection.roulette,
    "universal" : selection.universal,
    "ranking" : selection.ranking,
    "boltzmann" : selection.boltzmann,
    "deterministic_tournament" : selection.deterministic_tournament,
    "probabilistic_tournament" : selection.probabilistic_tournament,
}

REPLACE_MAP = {
    "traditional": replacement.traditional,
    "young_bias": replacement.young_bias,
    "generational_gap": replacement.generational_gap,
}

STOPPING_MAP = {
    "quantity" : None
}

PAIRING_MAP = {
    "staggered": pairing.staggered,
}

MUTATION_FUNCTION_MAP = {
    "uniform": mutation_func.uniform,
    "random": mutation_func.randomize,
    "decrease": mutation_func.decrease,
    "increase": mutation_func.increase
}

populations_list = []
iterations = 0


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

def max_iterations(max_iterations):
    global iterations
    return iterations >= max_iterations




def iterate(population, config):
    player_class = CLASS_MAP[config["class"]]
    children_count = config["children"]
    stopping_condition = partial(max_iterations, config["max_iterations"])
    pair_genotypes = PAIRING_MAP[config["pairing"]]
    crossover = CROSSOVER_MAP[config["crossover"]]
    mutate = MUTATION_MAP[config["mutation_type"]]
    mutation_rate = config["mutation_rate"]
    mutation_function = MUTATION_FUNCTION_MAP[config["mutation_function"]]
    replace = REPLACE_MAP[config["replace"]]
    select1 = SELECTION_MAP[config["selection1"]]
    select2 = SELECTION_MAP[config["selection2"]]
    select3 = SELECTION_MAP[config["selection3"]]
    select4 = SELECTION_MAP[config["selection4"]]
    global iterations
    iterations = 0
    while not stopping_condition():
        iterations += 1
        parents = select1(population, children_count) # TODO: Consider selection2
        children_genotypes = pair_genotypes([p.genotype for p in parents], crossover)
        children = [Player(player_class, mutate(genotype, mutation_function(mutation_rate, generation=iterations),  PLAYER_GENE_DOMAINS)) for genotype in children_genotypes]
        population = replace(population, children, select3)
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