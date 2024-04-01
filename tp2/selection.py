import random
def composite(population, sample_size, selection_method1, selection_method2, coef_method1):
    sample_size1 = int(coef_method1 * sample_size)
    
    parents1 = selection_method1(population, sample_size1)
    parents2 = selection_method2(population, sample_size - sample_size1)
    return parents1 + parents2


def elite(population, sample_size):
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:sample_size]


def roulette(population, sample_size, pseudo_fitness_function = lambda p: p.fitness, random_function = lambda x,y: x):
    selection = []
    sum_fitness = sum([pseudo_fitness_function(p) for p in population]) 
    relative_fitness = [p.fitness / sum_fitness for p in population]    
    accumulated_fitness = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]
    for i in range(sample_size):
        r = random_function(random.uniform(0, 1), i)
        for j in len(population):
            if r < accumulated_fitness[j]:
                selection.append(population[j])
                break
    return selection

def universal(population, sample_size):
    return roulette(population, sample_size, random_function=lambda x,y: (x+y)/sample_size)

def ranking(population, sample_size):
    population_copy = population.copy()
    population_copy.sort(key=lambda x: x.fitness, reverse=True)
    return roulette(population_copy, sample_size, lambda p: 1 - population.index(p) / len(population))
    

def boltzmann(population, sample_size):
    # TODO
    pass
    

def deterministic_tournament(population, sample_size, random_pick_size):
    selection = []
    for _ in range(sample_size):
        sample = random.sample(population, random_pick_size)
        selection.append(max(sample, key=lambda x: x.fitness))
    return selection
    
    
def probabilistic_tournament(population, sample_size):
    threshold = random.uniform(0.5, 1)
    selection = []
    while len(selection) < sample_size:
        r = random.uniform(0, 1)
        sample = random.sample(population, 2) 
        element = max(sample, key=lambda x: x.fitness) if r < threshold else min(sample, key=lambda x: x.fitness)
        selection.append(element)
    return selection






