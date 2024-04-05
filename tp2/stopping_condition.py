from math import isclose

def max_generations(max_generations, iterations, population_list):
    return iterations >= max_generations 

def get_population_bag(population):
    bag = {}
    for p in population:
        bag[p.genotype] = bag.get(p.genotype, 0) + 1
    return bag

def get_intersection_bag(bag1, bag2):
    count = 0
    intersection_bag = {}
    for k in bag1.keys():
        min_count = min(bag1[k], bag2.get(k, 0))
        if min_count > 0:
            count += min_count 
            intersection_bag[k] = min_count 
    return intersection_bag, count
    

def structure(max_generations, iterations, population_list):
    if iterations < max_generations:
        return False

    intersection_bag = get_population_bag(population_list[-1])
    for i in range(2,max_generations):
        current_bag = get_population_bag(population_list[-i])
        intersection_bag, count = get_intersection_bag(intersection_bag, current_bag)

        if count/len(population_list[0]) < 0.95: # TODO: Consider a parameter
            return False
    return True


def content(max_generations, iterations, population_list):
    global last_max
    global max_streak 
    
    current_max = max(population_list[-1])
    if isclose(current_max, last_max, rel_tol=1e-8):
        max_streak += 1
    else:
        last_max = current_max
        max_streak = 0

    return max_generations <= max_streak

def around_optimus_prime(threshold, iterations, population_list):
    return max(population_list[-1]) >= threshold
