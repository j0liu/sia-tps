from math import isclose
last_max = None

def max_generations(iterations, population_list, params):
    max_generations = params['max_generations']
    return iterations >= max_generations 

def _get_population_bag(population):
    bag = {}
    for p in population:
        bag[p] = bag.get(p, 0) + 1
    return bag

def _get_intersection_bag(bag1, bag2):
    count = 0
    intersection_bag = {}
    for k in bag1.keys():
        min_count = min(bag1[k], bag2.get(k, 0))
        if min_count > 0:
            count += min_count 
            intersection_bag[k] = min_count 
    return intersection_bag, count
    

def structure(iterations, population_list, params):
    streak_length = params['streak_length']
    similarity = params['similarity']
    if iterations < streak_length:
        return False

    intersection_bag = _get_population_bag(population_list[-1])
    for i in range(2,streak_length):
        current_bag = _get_population_bag(population_list[-i])
        intersection_bag, count = _get_intersection_bag(intersection_bag, current_bag)

        if count/len(population_list[0]) < similarity: # TODO: Consider a parameter
            return False
    return True


def content(iterations, population_list, params):
    global last_max
    global current_streak 
    streak_length = params['streak_length']
    
    current_max = max(p.fitness for p in population_list[-1])
    if len(population_list) > 1 and last_max != None and isclose(current_max, last_max, rel_tol=1e-8):
        current_streak += 1
    else:
        last_max = current_max
        current_streak = 0

    return streak_length <= current_streak

def around_optimus_prime(iterations, population_list, params):
    threshold = params['threshold']
    return max(p.fitness for p in population_list[-1]) >= threshold
