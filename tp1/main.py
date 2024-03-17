import numpy as np
from enum import Enum

class Direction(object):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Entity(object):
    SPACE = 0
    WALL = 1
    BOX = 2
    PLAYER = 3
    GOAL = 4
    BOX_ON_GOAL = GOAL + BOX
    PLAYER_ON_GOAL = GOAL + PLAYER

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
        #self.goals = self.find_goals()
        #self.boxes = self.find_boxes()
    
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

    def move(self, action):
        new_matrix = self.matrix.copy()
        directions = {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.LEFT: (0, -1),
            Direction.RIGHT: (0, 1)
        }
        dir = directions[action] 
        new_matrix[self.player] = Entity.SPACE if new_matrix[self.player] == Entity.PLAYER else Entity.GOAL
        pos = sum_tuples(self.player, dir) 
        next_pos = sum_tuples(pos, dir)

        if has_box(new_matrix, pos) and can_move(new_matrix, next_pos):
            move_box(new_matrix, pos, next_pos)
        if can_move(new_matrix, pos):
            move_player(new_matrix, self.player, pos)
            return SokobanState(new_matrix)
        else:
            return None


class Node:
    def __init__(self, state, parent, action, cost) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.children = []
    
    def expand(self):
        pass


def parse_sokoban_level(level_file):
    # Define the symbols

    symbols = {
        '#': Entity.WALL,
        ' ': Entity.SPACE,
        '.': Entity.GOAL,
        '$': Entity.BOX,
        '@': Entity.PLAYER,
        '*': Entity.BOX_ON_GOAL,
        '+': Entity.PLAYER_ON_GOAL
    }

    # Split the level string into rows
    with open (level_file, "r") as file:
        level_string = file.read()

    rows = level_string.split('\n')

    # Parse the level
    level = []
    for row in rows:
        level_row = []
        for char in row:
            if char not in symbols:
                raise ValueError(f"Invalid character '{char}' in level string")
            level_row.append(symbols.get(char))  # Use 'unknown' for any unexpected character
        level.append(level_row)
    
    return level



def main():
    # Your Sokoban level
    sokoban_level = "tp1/levels/lvl1.txt"

    # Parse the level
    parsed_level = parse_sokoban_level(sokoban_level)
    initial_matrix = np.matrix(parsed_level)
    print(initial_matrix)


    initial_state = SokobanState(initial_matrix)
    print(initial_state.player)
    

if __name__ == "__main__":
    main()

