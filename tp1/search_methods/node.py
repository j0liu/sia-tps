from abc import ABC, abstractmethod

class Node(ABC):
    def __init__(self, state, parent, action, cost, comparator = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.children = []
        self.comparator = comparator

    @abstractmethod
    def expand(self):
        pass

    def f(self):
        return self.cost + self.h
    
    def __lt__(self, other):
        return self.cost < other.cost if self.comparator is None else self.comparator(self, other)

    def __eq__(self, other):
        return self.state == other.state