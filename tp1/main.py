import numpy as np
import json
from sokoban import (
    SokobanState, Entity, SokobanNode, load_sokoban_config
)
from heuristics import (
    d_heuristic, md_heuristic, dp_heuristic, mdp_heuristic,
    manhattan_heuristic, euclidean_heuristic, max_heuristic, 
    manhattan_distance, euclidean_distance
)
from search_methods import bfs, dfs, greedy, astar
from map_parser import parse_sokoban_level
from functools import partial
from datetime import datetime

methods = {
    "dfs": dfs.search,
    "bfs": bfs.search,
    "greedy": greedy.search,
    "A*": astar.search
}

heuristics = {
    "euclidean": partial(d_heuristic, distance_function=euclidean_distance),
    "euclidean+p": euclidean_heuristic, 
    "manhattan": partial(d_heuristic, distance_function=manhattan_distance),
    "manhattan+p": manhattan_heuristic,
    "mod_manhattan+p": partial(mdp_heuristic, distance_function=manhattan_distance),
    "max_manhattan+p": max_heuristic(manhattan_heuristic, partial(mdp_heuristic, distance_function=manhattan_distance))
}


def single():
    with open("tp1/sokoban_config.json") as f:
        config = json.load(f)

    SOKOBAN_LEVEL = config["level"] 
    search = methods[config["method"]]
    heuristic = heuristics.get(config.get("heuristic", None), None)
    load_sokoban_config(config)
    print(config)

    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)

    initial_matrix = np.matrix(parsed_level)
    initial_state = SokobanState.from_matrix(initial_matrix)
    print(initial_state)

    #Run it 10 times to get the time, visited, border and cost
    initial_node = SokobanNode(initial_state, None, None, 0)

    begin_time = datetime.now()
    (solution, visited, border) = search(initial_node, h=heuristic)
    finish_time = datetime.now()

    print(f"cost = {solution.cost} ")
    print(f"solution length = {len(solution.get_sequence())}")
    print(f"visited nodes = {len(visited)}")
    print(f"border = {len(border)}")
    print("Time = ", finish_time - begin_time)
    print("Solution = ", solution.get_sequence())

def test_multiple():
    with open("tp1/sokoban_config.json") as f:
        config = json.load(f)

    SOKOBAN_LEVEL = config["level"] 
    search = methods[config["method"]]
    heuristic = heuristics.get(config.get("heuristic", None), None)
    load_sokoban_config(config)
    print(config)

    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)

    initial_matrix = np.matrix(parsed_level)
    initial_state = SokobanState.from_matrix(initial_matrix)
    print(initial_state)

    results = {}
    cases = ["NO OPTIMIZATION", "OPTIMIZED"]
    #Run it 10 times to get the time, visited, border and cost
    for case in cases: 
        partial_results = {}
        for method in methods:
            method_results = {"time": []}
            search = methods[method]

            for _ in range(10):
                initial_node = SokobanNode(initial_state, None, None, 0)

                begin_time = datetime.now()
                (solution, visited, border) = search(initial_node, h=heuristic)
                finish_time = datetime.now()

                print(f"cost = {solution.cost} ")
                method_results["cost"] = solution.cost
                print(f"solution length = {len(solution.get_sequence())}")
                method_results["solution_length"] = len(solution.get_sequence())
                print(f"visited nodes = {len(visited)}")
                method_results["visited_nodes"] = len(visited)
                print(f"border = {len(border)}")
                method_results["border"] = len(border)
                print("Time = ", finish_time - begin_time)
                method_results["time"].append(str(finish_time - begin_time))

            partial_results[method] = method_results        

        results[case] = partial_results
        config["optimizations"]["corners"] = not config["optimizations"]["corners"] 
        config["optimizations"]["axis"] = not config["optimizations"]["axis"] 
        config["optimizations"]["dual"] = not config["optimizations"]["dual"] 
        load_sokoban_config(config)

    with open("tp1/results.json", "w") as f:
        json.dump(results, f)


def test_heuristics():
    with open("tp1/sokoban_config.json") as f:
        config = json.load(f)

    SOKOBAN_LEVEL = config["level"] 
    search = methods[config["method"]]
    load_sokoban_config(config)
    print(config)

    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)

    initial_matrix = np.matrix(parsed_level)
    initial_state = SokobanState.from_matrix(initial_matrix)
    print(initial_state)

    TRIES = 10
    results = {}
    #Run it 10 times to get the time, visited, border and cost
    for heuristic in heuristics:
        search = methods[config["method"]]
        heuristic_function = heuristics[heuristic]
        heuristic_result = {"time": []}

        for _ in range(TRIES):
            initial_node = SokobanNode(initial_state, None, None, 0)

            begin_time = datetime.now()
            (solution, visited, border) = search(initial_node, h=heuristic_function)
            finish_time = datetime.now()

            print(f"cost = {solution.cost} ")
            heuristic_result["cost"] = solution.cost
            print(f"solution length = {len(solution.get_sequence())}")
            heuristic_result["solution_length"] = len(solution.get_sequence())
            print(f"visited nodes = {len(visited)}")
            heuristic_result["visited_nodes"] = len(visited)
            print(f"border = {len(border)}")
            heuristic_result["border"] = len(border)
            print("Time = ", finish_time - begin_time)
            heuristic_result["time"].append(str(finish_time - begin_time))
        
        results[heuristic] = heuristic_result


    with open("tp1/heuristic_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    # single()
    # test_multiple()
    test_heuristics()
