import numpy as np
from utils import normalize

class PlayerClass(object):
    WARRIOR = (0.6, 0.4)
    ARCHER = (0.9, 0.4)
    DEFENDER = (0.1, 0.9)
    INFILTRATE = (0.8, 0.3)

class ATTRIBUTES(object):
    HEIGHT = 0
    STR = 1
    AGI = 2
    EXP = 3
    RES = 4
    HEALTH = 5

class GeneDomain:
    ATTRIBUTE = (0, 150)
    HEIGHT = (1.3, 2.0)

PLAYER_GENE_DOMAINS = [GeneDomain.HEIGHT, GeneDomain.ATTRIBUTE, GeneDomain.ATTRIBUTE, GeneDomain.ATTRIBUTE, GeneDomain.ATTRIBUTE, GeneDomain.ATTRIBUTE]

class Player(object):
    def __init__(self, player_class, genotype) -> None:
        self.atk_coef, self.def_coef = player_class

        if genotype[0] < 1.3 or genotype[0] > 2:
            raise ValueError("Invalid height value")

        if any([i < 0 for i in genotype[1:]]): 
            raise ValueError("Invalid items")

        self.genotype = genotype if sum(genotype[1:]) == 150 else np.concatenate((genotype[0:1], normalize(genotype[1:], 150)))
        
        self.h, self.str_items, self.agi_items, self.exp_items, self.res_items, self.health_items = genotype 
    
        self.fitness = self.atk_coef*self.attack() + self.def_coef*self.defense()


    def get_genotype(self):
        return self.genotype 

    def strength_p(self):
        return 100 * np.tanh(0.01 * self.str_items)

    def agility_p(self):
        return np.tanh(0.01 * self.agi_items)

    def experience_p(self):
        return 0.6 * np.tanh(0.01 * self.exp_items)

    def resistence_p(self):
        return np.tanh(0.01 * self.res_items)

    def health_p(self):
        return 100 * np.tanh(0.01 * self.health_items)
  
    def atm(self):
        return 0.5 - (3 * self.h - 5) ** 4 + (3 * self.h - 5) ** 2 + self.h / 2
  
    def dem(self):
        return 2 + (3 * self.h - 5) ** 4 - (3 * self.h - 5) ** 2 - self.h / 2

    def attack(self):
        return (self.agility_p() + self.experience_p()) * self.strength_p() * self.atm()
  
    def defense(self):
        return (self.resistence_p() + self.experience_p()) * self.health_p() * self.dem()
    
    def __str__(self):
        return f"Player: h:{self.h}, s{self.str_items}, a{self.agi_items}, e{self.exp_items}, r{self.res_items}, hp{self.health_items}, fitness:{self.fitness}"
    
    def __repr__(self):
        return self.__str__()