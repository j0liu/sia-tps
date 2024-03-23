import numpy as np
from search_methods.node import Node
import json
import sys
import collections

class Direction(object):
    UP = '↑'
    DOWN = '↓'
    LEFT = '←'
    RIGHT = '→'

class Versor(object):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

directions = {
    Direction.UP: Versor.UP,
    Direction.DOWN: Versor.DOWN,
    Direction.LEFT: Versor.LEFT,
    Direction.RIGHT: Versor.RIGHT
}

class Entity(object):
    SPACE = ' '
    WALL = '#'
    BOX = '▧'
    PLAYER = '☺'
    GOAL = '!'
    BOX_ON_GOAL = '★'
    PLAYER_ON_GOAL = 'x'


with open("tp1/sokoban_config.json") as f:
    config = json.load(f)
OPT_CORNER =config["optimizations"]["corners"]
OPT_AXIS = config["optimizations"]["axis"]
DUAL = config["optimizations"]["dual"]


def perp(v):
    return (-v[1], v[0])

def opose(v):
    return (-v[0], -v[1])

def sum_tuples(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def has_goal(matrix, pos):
    return matrix[pos] == Entity.GOAL or matrix[pos] == Entity.BOX_ON_GOAL or matrix[pos] == Entity.PLAYER_ON_GOAL

def has_box(matrix, pos):
    return matrix[pos] == Entity.BOX or matrix[pos] == Entity.BOX_ON_GOAL

def has_wall(matrix, pos):
    return matrix[pos] == Entity.WALL

def can_move(matrix, pos):
    return pos[0] >= 0 and pos[0] < matrix.shape[0] and pos[1] >= 0 and pos[1] < matrix.shape[1] and matrix[pos] != Entity.WALL and not has_box(matrix, pos) 


def move_box(matrix, old, new):
    matrix[old] = Entity.SPACE if matrix[old] == Entity.BOX else Entity.GOAL
    matrix[new] = Entity.BOX if matrix[new] == Entity.SPACE else Entity.BOX_ON_GOAL

def move_player(matrix, old, new):
    matrix[old] = Entity.SPACE if matrix[old] == Entity.PLAYER else Entity.GOAL
    matrix[new] = Entity.PLAYER if matrix[new] == Entity.SPACE else Entity.PLAYER_ON_GOAL

def verify_dead_state(matrix, box):
    # After this if, it is guaranteed that the box is not in a goal, and that there is at least one wall around it 
    if matrix[box] == Entity.BOX_ON_GOAL or all(matrix[pos] != Entity.WALL for pos in [sum_tuples(box, dir) for dir in [Versor.UP, Versor.DOWN, Versor.LEFT, Versor.RIGHT]]):
        return False

    if (OPT_CORNER):
        # ((!up or !down) and (!left or !right)) : corner
        if ((has_wall(matrix, sum_tuples(box,Versor.UP)) or has_wall(matrix, sum_tuples(box, Versor.DOWN))) and 
            (has_wall(matrix, sum_tuples(box, Versor.RIGHT)) or has_wall(matrix, sum_tuples(box, Versor.LEFT)))):
            return True
    
    if (OPT_AXIS):
        # If there is a wall in the x-axis (left or right), the box can only move in the y-axis, and vice versa
        (dir, wall_dir1, wall_dir2) = (Versor.UP, Versor.LEFT, Versor.RIGHT) if has_wall(matrix, sum_tuples(box,Versor.LEFT)) or has_wall(matrix, sum_tuples(box,Versor.RIGHT)) else (Versor.LEFT, Versor.UP, Versor.DOWN)

        dead_ends = 0
        for d in [dir, opose(dir)]: 
            pos = box 
            while has_wall(matrix, sum_tuples(pos, wall_dir1)) or has_wall(matrix, sum_tuples(pos, wall_dir2)):
                pos = sum_tuples(pos, d)
                if matrix[pos] in [Entity.GOAL, Entity.PLAYER_ON_GOAL]:
                    return False

                if has_wall(matrix, pos):
                    dead_ends += 1
                    break

        if dead_ends == 2:
            return True
    
    if (DUAL):
        p_neighbor = [(sum_tuples(box, dir), dir) for dir in [Versor.UP, Versor.DOWN, Versor.LEFT, Versor.RIGHT] if has_box(matrix, sum_tuples(box, dir))]
        for (ne, dir) in p_neighbor:
            if (has_wall(matrix, sum_tuples(box, perp(dir))) or has_wall(matrix, sum_tuples(box, opose(perp(dir))))) or (
                has_wall(matrix, sum_tuples(ne, perp(dir))) and has_wall(matrix, sum_tuples(ne, opose(perp(dir))))):
                return True

    return False


class SokobanState(object):
    def __init__(self, matrix, goals, boxes, boxes_on_goal, player) -> None:
        self.matrix = matrix
        self.goals = goals
        self.boxes = boxes
        self.boxes_on_goal = boxes_on_goal
        self.player = player
    
    @classmethod
    def from_matrix(cls, matrix):
        goals = []
        boxes = []
        boxes_on_goal = []
        player = None
        for t, cell in np.ndenumerate(matrix):
            if cell == Entity.PLAYER or cell == Entity.PLAYER_ON_GOAL:
                player = t
            if cell == Entity.GOAL or cell == Entity.PLAYER_ON_GOAL:
                goals.append(t)
            elif cell == Entity.BOX:
                boxes.append(t)
            elif cell == Entity.BOX_ON_GOAL:
                boxes_on_goal.append(t)
        if player is None:
            raise ValueError("Player not found in matrix")
        boxes = np.array(boxes)
        goals = np.array(goals)
        boxes_on_goal = np.array(boxes_on_goal)
        return cls(matrix, goals, boxes, boxes_on_goal, player)


    def __eq__(self, other):
        return self.player == other.player and np.array_equal(self.boxes, other.boxes)

    def __str__(self):
        if self is not None:
            representation = ""
            for row in self.matrix:
                for _, cell in np.ndenumerate(row):
                    representation += cell
                representation += "\n"
            return representation
        return None


    def is_solution(self):
        for _, cell in np.ndenumerate(self.matrix):
                if cell == Entity.GOAL or cell == Entity.PLAYER_ON_GOAL or cell == Entity.BOX:
                    return False
        return True

    def move(self, action, ignore_dead_states=False):
        new_matrix = self.matrix.copy()
        dir = directions[action] 
        pos = sum_tuples(self.player, dir) 
        next_pos = sum_tuples(pos, dir)

        if has_box(new_matrix, pos) and can_move(new_matrix, next_pos):
            move_box(new_matrix, pos, next_pos)

            if not ignore_dead_states and verify_dead_state(new_matrix, next_pos):
                return None
            
        if can_move(new_matrix, pos):
            move_player(new_matrix, self.player, pos)
            return SokobanState.from_matrix(new_matrix)
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