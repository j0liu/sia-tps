def search(initial_node, h=None):
    border = [initial_node]
    border_set = set()
    visited = set() 
    border_set.add(initial_node)
    while len(border) > 0:
        current = border.pop(0)
        border_set.remove(current)

        if current.state.is_solution():
            return (current, visited, border)

        visited.add(current)

        new_nodes = [x for x in current.expand() if x not in visited and x not in border_set]
        border.extend(new_nodes)
        border_set.update(new_nodes)
    return (None, visited, border)
