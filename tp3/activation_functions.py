import numpy as np
import math

def step(x):
    return 1 if x >= 0 else -1

def id(x):
    return x

def one(x):
    return 1

def tanh(x, beta):
    return math.tanh(beta * x)

def tanh_derivative(x, beta):
    return beta * (1 - math.tanh(x)**2)

def logistic(x, beta):
    return 1 / (1 + math.exp(-2*beta*x))

def logistic_derivative(x, beta):
    return 2 * beta * logistic(x, beta) * (1 - logistic(x, beta))

#[x_min, x_max] -> [a,b]
def normalize(x, x_min, x_max, a, b):
    return (b-a) * (x - x_min) / (x_max - x_min) + a

#[a,b] -> [x_min, x_max]
def denormalize(x, x_min, x_max, a, b):
    return (x - a) * (x_max - x_min) / (b-a) + x_min

def scale(x, dom_min, dom_max, cod_min, cod_max):
    return (x - dom_min) * (cod_max - cod_min) / (dom_max - dom_min) + cod_min