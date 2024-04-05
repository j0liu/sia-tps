import mutation
import replacement
import crossover
import pairing
import mutation_func
import stopping_condition as sc
import selection
from player import PlayerClass

CLASS_MAP = {
    "warrior": PlayerClass.WARRIOR,
    "archer": PlayerClass.ARCHER,
    "defender": PlayerClass.DEFENDER,
    "infiltrate": PlayerClass.INFILTRATE
}

CROSSOVER_MAP = {
    "onepoint": crossover.one_point,
    "twopoint": crossover.two_point,
    "anular": crossover.anular,
    "uniform": crossover.uniform,
}

MUTATION_MAP = {
  "gene": mutation.mutate_gene,
  "multigene": mutation.mutate_multigene,
#   "multigene_limited": mutation.mutate_multigene_limited,
  "none": mutation.mutate_none
}

MUTATION_FUNCTION_MAP = {
    "uniform": mutation_func.uniform,
    "random": mutation_func.randomize,
    "decrease": mutation_func.decrease,
    "increase": mutation_func.increase
}

SELECTION_MAP = {
    "elite" : selection.elite,
    "roulette" : selection.roulette,
    "universal" : selection.universal,
    "ranking" : selection.ranking,
    "boltzmann" : selection.boltzmann,
    "deterministic_tournament" : selection.deterministic_tournament,
    "probabilistic_tournament" : selection.probabilistic_tournament,
}

REPLACE_MAP = {
    "traditional": replacement.traditional,
    "young_bias": replacement.young_bias
}

STOPPING_MAP = {
    "max_generations" : sc.max_generations, 
    "structure" : sc.structure,
    "content" : sc.content,
    "around_optimal" : sc.around_optimus_prime
}

PAIRING_MAP = {
    "staggered": pairing.staggered,
    "inverted": pairing.inverted,
    "parallel": pairing.parallel
}

