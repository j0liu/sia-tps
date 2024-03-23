from map_parser import parse_sokoban_level
from sokoban import SokobanState
import numpy as np
import json

def visualize(initial_state, sequence):
    state = initial_state
    print(f"Initial state ({len(sequence)} steps in total)")
    print(initial_state.matrix)
    for i, step in enumerate(sequence):
        input()
        state = state.move(step)
        print(f"Step {i+1}: {step}")
        print(state.matrix)
    print("fin!!!")
    input()

if __name__ == "__main__":
    with open("tp1/sokoban_config.json") as f:
        config = json.load(f)
    SOKOBAN_LEVEL = config["level"] 
    parsed_level = parse_sokoban_level(SOKOBAN_LEVEL)
    initial_matrix = np.matrix(parsed_level)
    sequence = ['←', '→', '→', '←', '↑', '↑', '←', '←', '↓', '←', '↓', '↓', '→', '↑', '↑', '↑', '↑', '↓', '→', '→', '→', '→', '↓', '→', '↓', '↓', '←', '↑', '↑', '↑', '↑', '↓', '↓', '↓', '→', '→', '→', '→', '↑', '↑', '←', '↑', '←', '↑', '←', '←', '←', '→', '↓', '↓', '←', '←', '↑', '↑', '↑', '↓', '↓', '↓', '↓', '↓', '←', '←', '←', '←', '←', '←', '↑', '↑', '→', '↑', '→', '↑', '→', '→', '→', '←', '↓', '↓', '→', '→', '↑', '↑']
    visualize(SokobanState(initial_matrix), sequence)

