from sokoban import Entity

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
            level_row.append(symbols.get(char)) 
        level.append(level_row)
    
    return level

