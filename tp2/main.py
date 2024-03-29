import json
import random
from player import Player, PlayerClass
from functools import partial


CLASS_MAP = {
    "warrior": PlayerClass.WARRIOR,
    "archer": PlayerClass.ARCHER,
    "defender": PlayerClass.DEFENDER,
    "infiltrate": PlayerClass.INFILTRATE
}

CROSSOVER_MAP = {
    "one-point": None
}

MUTATION_MAP = {
  "gene": None,
  "multigene": None,
  "uniform": None
}

SELECTION_MAP = {
    "elite" : None
}

REPLACE_MAP = {
    "traditional": None
}

STOPPING_MAP = {
    "quantity" : None
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
        p = Player(player_class, h, tuple(stats))
        population.append(p)
    return population

def max_iterations(max_iterations, current_iteration):
    return current_iteration >= max_iterations


def iterate(population, stopping_condition, crossover, mutation, selection1, selection2, replace1, replace2):
    while not stopping_condition():
        # Evaluate
        # Selection
        # Crossover
        # Mutation
        # Replace
        pass 


def main():
    with open("tp2/config.json") as config:
        config = json.load(config)

    POPULATION_SIZE = config["initial_population_size"] 
    PLAYER_CLASS = CLASS_MAP[config["class"]]

    population = generate_population(POPULATION_SIZE, PLAYER_CLASS)
    # print(population)
    iterate(population, 
            partial(max_iterations, config["max_iterations"]),
            CROSSOVER_MAP[config["crossover"]],
            SELECTION_MAP[config["selection1"]], SELECTION_MAP[config["selection2"]], 
            REPLACE_MAP[config["replace1"]], REPLACE_MAP[config["replace2"]], 
            MUTATION_MAP[config["mutation"]])





if __name__ == "__main__":
    main()