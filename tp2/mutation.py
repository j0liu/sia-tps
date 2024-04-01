import random

def mutate_gene(genotype, mutation_rate, gene_range):
    genotype_copy = genotype.copy()

    if random.random() < mutation_rate:
        index = random.randint(0, len(genotype)-1)
        get_random = random.randint if type(gene_range[index][0]) == int else random.uniform
        genotype_copy[index] = get_random(gene_range[index][0], gene_range[index][1]) 
    
    return genotype_copy

def mutate_multigene_limited(genotype, mutation_rate, gene_range):
    # TODO
    pass
    
def mutate_multigene_uniform(genotype, mutation_rate, gene_range):
    # TODO
    pass

def mutate_complete(genotype, mutation_rate, gene_range):
    # TODO
    pass

def mutate_none(genotype, mutation_rate, gene_range):
    return genotype
