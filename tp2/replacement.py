def traditional(population, children, select, iterations):
    return select(population + children, len(population), iterations)

def young_bias(population, children, select, iterations):
    children_len = len(children)
    population_len = len(population)
    return select(children, population_len, iterations) if children_len > population_len else children + select(population, population_len - children_len, iterations)
 