import numpy as np
import gudhi


def extract_epsilon_optimal_through_persistent_homology(correlation_matrix):
    """Return a correlation threshold based on the longest persistent feature."""

    # Compute the distance matrix from the correlation matrix
    distance_matrix = np.sqrt(2 * (1 - np.abs(correlation_matrix)))

    # Build a Rips complex and compute persistence
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=5)
    persistence = simplex_tree.persistence()

    # Select the death time of the longest finite interval
    best_death = None
    best_persistence = -np.inf
    for _, (birth, death) in persistence:
        if death == np.inf:
            continue
        persistence_length = death - birth
        if persistence_length > best_persistence:
            best_persistence = persistence_length
            best_death = death

    if best_death is None:
        return 0.0

    # Convert the death time back to a correlation value
    return 1 - (best_death ** 2) / 2
