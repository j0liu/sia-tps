def search(initial_node, h=None):
    border = [initial_node]
    visited = set()
    while len(border) > 0:
        current = border.pop()

        if current.state.is_solution():
            return (current, visited)

        visited.add(current)

        new_nodes = [x for x in current.expand() if x not in visited and x not in border]
        border.extend(new_nodes)
    return (None, visited)
