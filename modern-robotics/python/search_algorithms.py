


def a_star_algorithm(start_node, goal_node, edge_distances_matrix):
    """
    Implements the A* search algorithm to find the shortest path 
    between start_node and goal_node using a distance matrix.
    """
    open_list = []
    closed_list = set() # Using a set for O(1) lookup performance

    # Initialize start node costs
    start_node.set_past_cost(0)
    start_node.set_estimated_total_cost(start_node.get_hvalue())
    open_list.append(start_node)

    while len(open_list) > 0:
        # Pick the node with the lowest f(n) = g(n) + h(n)
        current_node = min(open_list, key=lambda node: node.get_estimated_total_cost())
        
        # Debugging print
        print(f"Current node: {current_node.get_id()}, Estimated Total Cost: {current_node.get_estimated_total_cost()}")

        # Check if we reached the goal
        if current_node.get_id() == goal_node.get_id():
            return reconstruct_path(current_node)
        
        open_list.remove(current_node)
        closed_list.add(current_node)

        for neighbor in current_node.get_neighbors():
            if neighbor in closed_list:
                continue

            # Calculate g(n): cost from start to current + edge weight to neighbor
            weight = edge_distances_matrix[current_node.get_id() - 1][neighbor.get_id() - 1]
            tentative_past_cost = current_node.get_past_cost() + weight

            # If we found a shorter path to this neighbor
            if tentative_past_cost < neighbor.get_past_cost():
                neighbor.set_parent(current_node)
                neighbor.set_past_cost(tentative_past_cost)
                # Update f(n) = g(n) + h(n)
                neighbor.set_estimated_total_cost(tentative_past_cost + neighbor.get_hvalue())
                
                if neighbor not in open_list:
                    open_list.append(neighbor)
    
    return None # No path found


def reconstruct_path(node):
    """
    Backtracks from the goal node to the start node using parent pointers.
    """
    path = []
    while node is not None:
        path.append(node)
        node = node.get_parent()
    return path[::-1] # Reverse the list to get Start -> Goal order


def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node)
        node = node.get_parent()
    return path[::-1] #Return reversed path

