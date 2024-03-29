import numpy as np

class PlayerClass(object):
    WARRIOR = (0.6, 0.4)
    ARCHER = (0.9, 0.4)
    DEFENDER = (0.1, 0.9)
    INFILTRATE = (0.8, 0.3)

class Player(object):
    def __init__(self, player_class, h, items_t) -> None:
        self.atk_coef, self.def_coef = player_class

        if h < 1.3 or h > 2:
            raise ValueError("Invalid height value")
        self.h = h

        if sum(items_t) != 150 or any([i < 0 for i in items_t]): 
            raise ValueError("Invalid items")
        
        self.str_items, self.agi_items, self.exp_items, self.res_items, self.health_items = items_t 
    
        self.fitness = self.atk_coef*self.attack() + self.def_coef*self.defense()


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
        return f"Player: {self.h}, {self.str_items}, {self.agi_items}, {self.exp_items}, {self.res_items}, {self.health_items}, {self.fitness}"
    
    def __repr__(self):
        return self.__str__()