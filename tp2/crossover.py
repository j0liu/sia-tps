import random
import numpy as np
import math

def one_point(genotype_1, genotype_2, params):
   locus = random.randint(0, len(genotype_1) - 1)    
   child_genotype_1 = np.concatenate((genotype_1[:locus], genotype_2[locus:]))
   child_genotype_2 = np.concatenate((genotype_2[:locus], genotype_1[locus:])) 
   return child_genotype_1, child_genotype_2

def two_point(genotype_1, genotype_2, params):
   locus_1 = random.randint(0, len(genotype_1) - 1)
   locus_2 = random.randint(0, len(genotype_1) - 1)
   if locus_1 > locus_2:
      locus_1, locus_2 = locus_2, locus_1
   child_genotype_1 = np.concatenate((genotype_1[:locus_1], genotype_2[locus_1:locus_2], genotype_1[locus_2:]))
   child_genotype_2 = np.concatenate((genotype_2[:locus_1], genotype_1[locus_1:locus_2], genotype_2[locus_2:]))
   return child_genotype_1, child_genotype_2

def anular(genotype_1, genotype_2, params):
   locus = random.randint(0, len(genotype_1) - 1)
   length = random.randint(0, math.ceil(len(genotype_1)/2))
   if locus + length >= len(genotype_1):
      start = (locus + length) % len(genotype_1)

      child_genotype_1 = np.concatenate((genotype_2[:start], genotype_1[start:locus], genotype_2[locus:]))
      child_genotype_2 = np.concatenate((genotype_1[:start], genotype_2[start:locus], genotype_1[locus:]))
   else:
      child_genotype_1 = np.concatenate((genotype_1[:locus], genotype_2[locus:locus + length], genotype_1[locus + length:]))
      child_genotype_2 = np.concatenate((genotype_2[:locus], genotype_1[locus:locus + length], genotype_2[locus + length:]))
   return child_genotype_1, child_genotype_2

def uniform(genotype_1, genotype_2, params):
   p = params['p']
   return non_uniform(genotype_1, genotype_2, lambda x: p)

def non_uniform(genotype_1, genotype_2, p_function, generation = 0):
   child_genotype_1 = np.empty(len(genotype_1))
   child_genotype_2 = np.empty(len(genotype_2))
   
   for i in range(len(genotype_1)):
      r = random.random()
      child_genotype_1[i], child_genotype_2[i] = (genotype_2[i], genotype_1[i]) if r < p_function(generation) else (genotype_1[i], genotype_2[i])
   return child_genotype_1, child_genotype_2