import numpy as np
from sokoban import SokobanState, Entity, SokobanNode, manhattan_heuristic, euclidean_heuristic

from search_methods.greedy import search


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
    sokoban_level = "tp1/levels/lvl4.txt"
    # Parse the level
    parsed_level = parse_sokoban_level(sokoban_level)
    initial_matrix = np.matrix(parsed_level)
    print(initial_matrix)

    initial_state = SokobanState(initial_matrix)
    print(initial_state.player)

    initial_node = SokobanNode(initial_state, None, None, 0)

    solution = search(initial_node, h=euclidean_heuristic)
    print(solution.state.matrix)
    print(solution.cost)
    print(solution.get_sequence())

if __name__ == "__main__":
    main()

