import numpy as np
from search_methods.node import Node

class Direction(object):
    UP = '↑'
    DOWN = '↓'
    LEFT = '←'
    RIGHT = '→'

class Entity(object):
    SPACE = ' '
    WALL = '#'
    BOX = 'b'
    PLAYER = 'p'
    GOAL = 'g'
    BOX_ON_GOAL = '!'
    PLAYER_ON_GOAL = 'x'

def sum_tuples(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def has_goal(matrix, pos):
    return matrix[pos] == Entity.GOAL or matrix[pos] == Entity.BOX_ON_GOAL or matrix[pos] == Entity.PLAYER_ON_GOAL

def has_box(matrix, pos):
    return matrix[pos] == Entity.BOX or matrix[pos] == Entity.BOX_ON_GOAL

def can_move(matrix, pos):
    return pos[0] >= 0 and pos[0] < matrix.shape[0] and pos[1] >= 0 and pos[1] < matrix.shape[1] and matrix[pos] != Entity.WALL and not has_box(matrix, pos) 

def move_box(matrix, old, new):
    matrix[old] = Entity.SPACE if matrix[old] == Entity.BOX else Entity.GOAL
    matrix[new] = Entity.BOX if matrix[new] == Entity.SPACE else Entity.BOX_ON_GOAL

def move_player(matrix, old, new):
    matrix[old] = Entity.SPACE if matrix[old] == Entity.PLAYER else Entity.GOAL
    matrix[new] = Entity.PLAYER if matrix[new] == Entity.SPACE else Entity.PLAYER_ON_GOAL

class SokobanState(object):
    def __init__(self, matrix) -> None:
        self.matrix = matrix
        self.player = self.find_player()
        # self.goals = self.find_goals()
        #self.boxes = self.find_boxes()
    
    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)

    def __print__(self):
        print(self.matrix)

    def find_player(self):
        for t, cell in np.ndenumerate(self.matrix):
            if cell == Entity.PLAYER or cell == Entity.PLAYER_ON_GOAL:
                return t

    
    def find_goals(self):
        goals = []
        for t, cell in np.ndenumerate(self.matrix):
                if cell == Entity.GOAL or cell == Entity.BOX_ON_GOAL:
                    goals.append(t)
        return goals
    
    def find_boxes(self):
        boxes = []
        for t, cell in np.ndenumerate(self.matrix):
            if cell == Entity.BOX or cell == Entity.BOX_ON_GOAL:
                boxes.append(t)
        return boxes

    def is_solution(self):
        for _, cell in np.ndenumerate(self.matrix):
                if cell == Entity.GOAL or cell == Entity.PLAYER_ON_GOAL or cell == Entity.BOX:
                    return False
        return True


    def move(self, action):
        new_matrix = self.matrix.copy()
        directions = {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1)
        }
        dir = directions[action] 
        pos = sum_tuples(self.player, dir) 
        next_pos = sum_tuples(pos, dir)

        if has_box(new_matrix, pos) and can_move(new_matrix, next_pos):
            move_box(new_matrix, pos, next_pos)
        if can_move(new_matrix, pos):
            move_player(new_matrix, self.player, pos)
            return SokobanState(new_matrix)
        return None


class SokobanNode(Node):
    def __init__(self, state, parent=None, action=None, cost=0):
        super().__init__(state, parent, action, cost)

    def expand(self):
        return [
            SokobanNode(self.state.move(action), self, action, self.cost + 1)
            for action in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
            if self.state.move(action) is not None  
        ]

    def __str__(self):
        return f"Action: {self.action}, Cost: {self.cost}" 
        # State: {self.state.matrix}"
    
    def get_sequence(self):
        sequence = []
        current = self
        while current is not None:
            sequence.insert(0, current.action)
            current = current.parent
        return sequence