import numpy as np
import json
from sokoban import (
    SokobanState, Entity, SokobanNode, 
    modified_distance_heuristic,
    manhattan_heuristic, euclidean_heuristic, max_heuristic, distance_heuristic,
    manhattan_distance
)
from search_methods import bfs, dfs, greedy, astar
from map_parser import parse_sokoban_level
from functools import partial

methods = {
    "dfs": dfs.search,
    "bfs": bfs.search,
    "greedy": greedy.search,
    "A*": astar.search
}

heuristics = {
    "manhattan": manhattan_heuristic,
    "euclidean": euclidean_heuristic,
    "max_eucliman": max_heuristic(manhattan_heuristic, euclidean_heuristic),
    "mod_manhattan": partial(modified_distance_heuristic, distance_function=manhattan_distance),
    "super_manhattan": max_heuristic(manhattan_heuristic, partial(modified_distance_heuristic, distance_function=manhattan_distance)),
}



def main():
    with open("tp1/sokoban_config.json") as f:
        config = json.load(f)

    SOKOBAN_LEVEL = config["level"] 
    search = methods[config["method"]]
    heuristic = heuristics.get(config.get("heuristic", None), None)

    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)

    initial_matrix = np.matrix(parsed_level)
    print(initial_matrix)

    initial_state = SokobanState(initial_matrix)
    print(initial_state.player)

    initial_node = SokobanNode(initial_state, None, None, 0)

    (solution, visited) = search(initial_node, h=heuristic)
    # (solution, visited) = search(initial_node)
    print(solution.state.matrix)
    print(solution.cost)
    print(solution.get_sequence())
    print(len(visited))
    print("fin!!!")

if __name__ == "__main__":
    main()

