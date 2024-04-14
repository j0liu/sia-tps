def staggered(parents_genotypes, crossover):
    children_genotypes = []
    for i in range(0, len(parents_genotypes), 2):
        g1 = parents_genotypes[i]
        g2 = parents_genotypes[i+1]
        children_genotypes.extend(crossover(g1, g2))
    return children_genotypes


def inverted(parents_genotypes, crossover):
    children_genotypes = []
    for i in range(0, int(len(parents_genotypes)/2)):
        g1 = parents_genotypes[i]
        g2 = parents_genotypes[-(i+1)]
        children_genotypes.extend(crossover(g1, g2))
    return children_genotypes

def parallel(parents_genotypes, crossover):
    children_genotypes = []
    middle = int(len(parents_genotypes)/2)
    for i in range(0, middle):
        g1 = parents_genotypes[i]
        g2 = parents_genotypes[i+middle]
        children_genotypes.extend(crossover(g1, g2))
    return children_genotypes

def complete(parents_genotypes, crossover):
    children_genotypes = []
    for i in range(0, len(parents_genotypes)):
        for j in range(i+1, len(parents_genotypes)):
            g1 = parents_genotypes[i]
            g2 = parents_genotypes[j]
            children_genotypes.extend(crossover(g1, g2))
    return children_genotypes

def alpha(parents_genotypes, crossover):
    children_genotypes = []
    for i in range(1, len(parents_genotypes)):
        g1 = parents_genotypes[1]
        g2 = parents_genotypes[i]
        children_genotypes.extend(crossover(g1, g2))
    return children_genotypes