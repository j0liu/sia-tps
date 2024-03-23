import heapq

def search(initial_node, h=None):
    border = [initial_node]
    initial_node.comparator = lambda n1, n2: h(n1) < h(n2)
    heapq.heappush(border, initial_node)

    visited = set()
    while len(border) > 0:
        current = heapq.heappop(border)

        if current.state.is_solution():
            return (current, visited, border)

        visited.add(current)

        new_nodes = [x for x in current.expand() if x not in visited and x not in border]
        for node in new_nodes:
            # node.h = h(node)
            heapq.heappush(border, node)

    return (None, visited, border)
