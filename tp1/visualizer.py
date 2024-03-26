from map_parser import parse_sokoban_level
from sokoban import SokobanState
import numpy as np
import json

def visualize(initial_state, sequence):
    state = initial_state
    print(f"Initial state ({len(sequence)} steps in total)")
    print(initial_state)
    for i, step in enumerate(sequence):
        input()
        state = state.move(step, ignore_dead_states=True)
        print(f"Step {i+1}: {step}")
        print(state)
    print("fin!!!")
    input()

if __name__ == "__main__":
    with open("sokoban_config.json") as f:
        config = json.load(f)
    SOKOBAN_LEVEL = config["level"] 
    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)
    initial_matrix = np.matrix(parsed_level)
    sequence = ['→', '→', '→', '↑', '↑', '↑', '←', '↑', '→', '→', '↓', '←', '↓', '↓', '↓', '→', '→', '↑', '←', '↓', '←', '↑', '↑', '↑', '←', '↑', '→', '↓', '→', '↑', '↑', '↑', '↑', '→', '↑', '←', '←', '←', '←', '→', '→', '→', '↓', '↓', '↓', '↓', '↓', '→', '↑', '↑', '↑', '↑', '←', '↑', '→', '→', '→', '↑', '→', '↓', '↓', '→', '↓', '←']
    visualize(SokobanState.from_matrix(initial_matrix), sequence)

