def traditional(population, children, select):
    return select(population + children, len(population))

def young_bias(population, children, select):
    children_len = len(children)
    population_len = len(population)
    return select(children, population_len) if children_len > population_len else children + select(population, population_len - children_len)
 