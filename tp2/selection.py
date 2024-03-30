def elite(population, sample_size):
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:sample_size]
