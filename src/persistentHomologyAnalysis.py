import numpy as np
import gudhi


def extract_epsilon_optimal_through_persistent_homology(correlation_matrix):
    """
    Extract the epsilon optimal value through persistent homology analysis
    of the given correlation matrix.

    :param correlation_matrix: 2D numpy array representing the correlation matrix.
    :return: The epsilon optimal value converted to a correlation value.
    """
    # Compute the distance matrix from the correlation matrix
    distance_matrix = np.sqrt(2 * (1 - np.abs(correlation_matrix)))

    # Create a Rips complex from the distance matrix
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=5)
    _ = simplex_tree.persistence()

    # List to collect the death times of simplices in various dimensions
    death_time_list = []

    # Iterate through dimensions to extract persistence intervals
    for dimension in range(5):
        intervals = simplex_tree.persistence_intervals_in_dimension(dimension)
        # Filter out infinite death times for dimension 0
        if dimension == 0:
            death_time_list.extend(
                [interval[1] for interval in intervals if interval[1] != np.inf]
            )
        else:
            death_time_list.extend([interval[1] for interval in intervals])

    # Convert death times to a numpy array
    death_times = np.array(death_time_list)

    # Find the death time closest to the median
    epsilon_optimal = death_times[
        np.abs(death_times - np.median(death_times)).argmin()
    ]

    # Convert the epsilon optimal value back to a correlation value
    corr_optimal = 1 - (epsilon_optimal ** 2) / 2

    return corr_optimal
