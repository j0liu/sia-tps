import random

def uniform(mutation_rate, generation):
    return mutation_rate

def randomize(mutation_rate, generation):
    return random.random()

def decrease(mutation_rate, generation):
    return 1/(generation+1)

def increase(mutation_rate, generation):
    return min(1,(generation+1)/100)

def oscilating_increase(mutation_rate, generation):
    return ((generation+1) % 100)/100

