import numpy as np
from sokoban import(
    can_move,
    sum_tuples,
    Versor
)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def distance_heuristic(node, distance_function):
    state = node.state
    player = state.player
    distance = 0
    for box in state.boxes:
        distance += min([distance_function(box, goal) for goal in state.goals])
    if len(state.boxes) > 0 :
        distance += min([distance_function(player, box) for box in node.state.boxes])
    return distance

def modified_distance_heuristic(node, distance_function):
    state = node.state
    player = state.player
    distance = 0
    sum_neighbors = lambda matrix, point: sum([ 1 if not can_move(matrix, sum_tuples(point, d)) else 0 for d in [Versor.UP, Versor.LEFT, Versor.RIGHT, Versor.DOWN]])
    for box in state.boxes:
        distance += min([distance_function(box, goal) + sum_neighbors(state.matrix, goal) for goal in state.goals])
        distance += sum_neighbors(state.matrix, box)
    if len(state.boxes) > 0:
        distance += min([distance_function(player, box) for box in node.state.boxes])
    return distance

def manhattan_heuristic(node):
    distance = distance_heuristic(node, manhattan_distance)
    return distance


def euclidean_heuristic(node):
    euclidian_distance = lambda a, b : np.linalg.norm(np.array(b) - np.array(a))
    distance = distance_heuristic(node, euclidian_distance)
    return distance


def max_heuristic(h1, h2):
    return lambda node: max(h1(node), h2(node))
