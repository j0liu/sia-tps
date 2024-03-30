import random
import numpy as np

def one_point_crossover(genotype_1, genotype_2):
   locus = random.randint(0, len(genotype_1) - 1)    
   child_genotype_1 = np.concatenate((genotype_1[:locus], genotype_2[locus:]))
   child_genotype_2 = np.concatenate((genotype_2[:locus], genotype_1[locus:])) 
   return child_genotype_1, child_genotype_2