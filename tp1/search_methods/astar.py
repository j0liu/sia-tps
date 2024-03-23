
def search(initial_node, h=None):
    border = [initial_node]
    visited = []
    while len(border) > 0:
        current = border.pop(0)

        if current.state.is_solution():
            return (current, visited, border)

        visited.append(current)

        new_nodes = [x for x in current.expand() if x not in visited and x not in border]
        for node in new_nodes:
            node.h = h(node)

        border.extend(new_nodes)
        border.sort(key=lambda x: (x.f(), x.h))

    return (None, visited, border)
