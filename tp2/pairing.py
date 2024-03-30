def staggered(parents_genotypes, crossover):
    children_genotypes = []
    for i in range(0, len(parents_genotypes), 2):
        g1 = parents_genotypes[i]
        g2 = parents_genotypes[i+1]
        children_genotypes.extend(crossover(g1, g2))
    return children_genotypes