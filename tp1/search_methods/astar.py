import heapq

def search(initial_node, h=None):
    border = []
    border_set = set()

    initial_node.h = h(initial_node)
    initial_node.comparator = lambda n1, n2: (n1.f(), n1.h) < (n2.f(), n2.h)

    heapq.heappush(border, initial_node)
    border_set.add(initial_node)

    visited = set()
    while len(border) > 0:
        current = heapq.heappop(border)
        border_set.remove(current)

        if current.state.is_solution():
            return (current, visited, border)

        visited.add(current)

        new_nodes = [x for x in current.expand() if x not in visited and x not in border_set]
        for node in new_nodes:
            node.h = h(node)
            heapq.heappush(border, node)
            border_set.add(node)

    return (None, visited, border)
