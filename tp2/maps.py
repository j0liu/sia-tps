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
    "infiltrate": PlayerClass.INFILTRATE,
    # "wizard": PlayerClass.WIZARD
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
    "increase": mutation_func.increase,
    "oscillating_increase": mutation_func.oscillating_increase
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
    "parallel": pairing.parallel,
    "complete": pairing.complete,
    "alpha": pairing.alpha
}

DETERMINISTIC_MAP = {
    "elite": {},
    "deterministic_tournament 1": {"random_pick_size": 2},
    "deterministic_tournament 2": {"random_pick_size": 4},
    "deterministic_tournament 3": {"random_pick_proportion": 0.1},
    "deterministic_tournament 4": {"random_pick_proportion": 0.2},
    "deterministic_tournament 5": {"random_pick_proportion": 0.5},
    "deterministic_tournament 7": {"random_pick_proportion": 0.9},
}

BOLTZMANN_MAP = {
    "elite": {},
    "boltzmann 1": {'t0': 100, 'tc': 1, 'k': 0.1},
    "boltzmann 2": {'t0': 500, 'tc': 50, 'k': 0.01},
    "boltzmann 3": {'t0': 10, 'tc': 1, 'k': 1},
    "boltzmann 4": {'t0': 1000, 'tc': 10, 'k': 1},
    "boltzmann 5": {'t0': 1000, 'tc': 100, 'k': 1},
    "boltzmann 6": {'t0': 1000, 'tc': 500, 'k': 1},
}

REPLACEMENT_PARENTS_MAP = {
    "4" : { "children" : 4 },
    "8" : { "children" : 8 },
    "10" : { "children" : 10 },
    "12" : { "children" : 12 },
    "16" : { "children" : 16 },
    "18" : { "children" : 18 },
    "20" : { "children" : 20 },
}
