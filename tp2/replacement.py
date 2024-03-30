def traditional(population, children, select):
    new_population = select(population + children, len(population))
    return select(new_population, len(population))