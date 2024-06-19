def build_simplicial_complex(vertices_neighbors):
    """
    Build a combinatorial complex from a dictionary of vertices and their neighbors.

    :param vertices_neighbors: Dictionary with vertices as keys and sets of neighbors as values.
    :return: Dictionary of cells where keys are frozensets of vertices and values are the ranks.
    """
    # Initialize the complex with 0-cells (individual vertices)
    simplicial_complex = {frozenset([vertex]): 0 for vertex in vertices_neighbors}

    # Start with 0-cells (vertices)
    current_cells = list(simplicial_complex.keys())
    rank = 1

    while True:
        new_cells = []
        
        # Discover new cells of the current rank by combining existing cells
        for cell in current_cells:
            # Find the common neighbors for all vertices in the current cell
            common_neighbors = set(vertices_neighbors[list(cell)[0]])
            for vertex in list(cell)[1:]:
                common_neighbors.intersection_update(vertices_neighbors[vertex])

            # Form new cells by adding each common neighbor to the current cell
            for neighbor in common_neighbors:
                if neighbor not in cell:
                    new_cell = frozenset(cell.union([neighbor]))
                    if new_cell not in simplicial_complex:
                        new_cells.append(new_cell)
                        simplicial_complex[new_cell] = rank

        # If no new cells are formed, stop the process
        if not new_cells:
            break

        # Update current cells and increment rank for the next iteration
        current_cells = new_cells
        rank += 1

    return simplicial_complex
