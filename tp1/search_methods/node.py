from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self, state, parent, action, cost) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.children = []

    @abstractmethod    
    def expand(self):
        pass

    def f(self):
        return self.cost + self.h

    def __eq__(self, other):
        return self.state == other.state