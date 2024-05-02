import numpy as np
import math

def step(x):
    return 1 if x >= 0 else -1

def id(x):
    return x

def one(x):
    return 1

def gen_tanh(beta):
    def tanh(x):
        return math.tanh(beta * x)
    return tanh

def gen_tanh_derivative(beta):
    def tanh_derivative(x):
        return beta * (1 - math.tanh(x)**2)
    return tanh_derivative

def gen_logistic(beta):
    def logistic(x):
        return 1 / (1 + math.exp(-beta*x))
    return logistic

def gen_logistic_derivative(beta):
    logistic = gen_logistic(beta)
    def logistic_derivative(x):
        return 2 * beta * logistic(x) * (1 - logistic(x))
    return logistic_derivative

#[x_min, x_max] -> [a,b]
def normalize(x, x_min, x_max, a, b):
    return (b-a) * (x - x_min) / (x_max - x_min) + a

#[a,b] -> [x_min, x_max]
def denormalize(x, x_min, x_max, a, b):
    return (x - a) * (x_max - x_min) / (b-a) + x_min

def scale(x, dom_min, dom_max, cod_min, cod_max):
    return (x - dom_min) * (cod_max - cod_min) / (dom_max - dom_min) + cod_min