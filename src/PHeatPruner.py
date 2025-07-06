from tqdm import tqdm
import numpy as np
import pandas as pd
from persistentHomologyAnalysis import extract_epsilon_optimal_through_persistent_homology
from simplicialComplexCreator import build_simplicial_complex


def PHeatPruner(X_train, X_test, sheafification=False):
    """Prune variables using persistent homology.

    ``X_train`` and ``X_test`` can be either 3D arrays representing
    ``(samples, variables, time)`` or regular 2D tabular data ``(samples, variables)``.

    When ``sheafification`` is ``True``, variance based features of connected
    variable sets are added back to the data.
    """

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    print(f"[Status] Correlating variables in the dataset with shape {X_train.shape}")

    if X_train.ndim == 3:
        variables = {f"Var{i}": X_train[:, i, :].reshape(-1) for i in range(X_train.shape[1])}
    elif X_train.ndim == 2:
        variables = {f"Var{i}": X_train[:, i] for i in range(X_train.shape[1])}
    else:
        raise ValueError("X_train must be a 2D or 3D array")

    df = pd.DataFrame(variables)
    corr_df = df.corr()
    corr = corr_df.values

    # Extract the epsilon threshold using persistent homology
    threshold = extract_epsilon_optimal_through_persistent_homology(corr)
    print(f"[Status] Selected correlation threshold {threshold}")

    print(f"[Status] Mapping variables into a tabular dataset")
    
    # Map data to a tabular DataFrame
    if X_train.ndim == 3:
        X_train_data_frame = {
            f"Var{i}_Time{j}": X_train[:, i, j]
            for i in range(X_train.shape[1])
            for j in range(X_train.shape[2])
        }
        X_test_data_frame = {
            f"Var{i}_Time{j}": X_test[:, i, j]
            for i in range(X_test.shape[1])
            for j in range(X_test.shape[2])
        }
    else:
        X_train_data_frame = {f"Var{i}": X_train[:, i] for i in range(X_train.shape[1])}
        X_test_data_frame = {f"Var{i}": X_test[:, i] for i in range(X_test.shape[1])}

    X_train_df = pd.DataFrame(X_train_data_frame)
    X_test_df = pd.DataFrame(X_test_data_frame)

    # Create mappings for variables to their respective columns in the DataFrame
    if X_train.ndim == 3:
        key_mappings = {f"Var{j}": [col for col in X_train_df.columns if f"Var{j}" in col] for j in range(X_train.shape[1])}
    else:
        key_mappings = {f"Var{j}": [f"Var{j}"] for j in range(X_train.shape[1])}

    # Initialize variables for the pruning process
    old_sum = X_train.shape[1] ** 2
    range_min = np.min(corr)
    range_max = np.max(corr)
    epsilon_steps = sorted(np.unique(np.abs(corr).flatten()))
    epsilon_step = np.min([
        epsilon_steps[i] - epsilon_steps[i - 1] for i in range(1, len(epsilon_steps))
    ])


    # Iterate through possible epsilon values to prune the dataset
    try:
        for epsilon in tqdm(np.arange(range_min - epsilon_step, range_max, epsilon_step)):
            significant_corr = np.abs(corr_df) > epsilon
            current_sum = significant_corr.sum().sum()

            if current_sum != old_sum:
                if np.isclose([epsilon],[threshold], atol=epsilon_step/2):
                    train_df_copy = X_train_df.copy()
                    test_df_copy = X_test_df.copy()

                    # Identify connected sets of variables
                    connected_set = {i: set(significant_corr.columns[significant_corr.loc[i]]) - {i} for i in significant_corr.index}
                    count_cut = 0

                    # Drop disconnected variables
                    for key, connections in connected_set.items():
                        if len(connections) == 0:
                            count_cut += 1
                            try:
                                train_df_copy.drop(key_mappings[key], axis=1, inplace=True)
                                test_df_copy.drop(key_mappings[key], axis=1, inplace=True)
                            except KeyError:
                                print(f"[Warning] No {key} in the columns")

                    print(f"[Status] Pruned {count_cut} variables based on connectivity!")

                    # Perform sheafification if required
                    if sheafification:
                        simplicial_complex = build_simplicial_complex(connected_set)
                        simplices = [i for i in simplicial_complex if simplicial_complex[i] > 0]

                        for simplex in tqdm(simplices):
                            simplex_cols = [key_mappings[vertex] for vertex in simplex]
                            train_supp_dict = {}
                            test_supp_dict = {}

                            if X_train.ndim == 3:
                                for time in range(len(simplex_cols[0])):
                                    name = "_".join([col[time] for col in simplex_cols])
                                    train_supp_dict[name] = [
                                        np.var([train_df_copy.loc[idx, col[time]] for col in simplex_cols])
                                        for idx in train_df_copy.index
                                    ]
                                    test_supp_dict[name] = [
                                        np.var([test_df_copy.loc[idx, col[time]] for col in simplex_cols])
                                        for idx in test_df_copy.index
                                    ]
                            else:
                                name = "_".join([col[0] for col in simplex_cols])
                                train_supp_dict[name] = train_df_copy[[col[0] for col in simplex_cols]].var(axis=1)
                                test_supp_dict[name] = test_df_copy[[col[0] for col in simplex_cols]].var(axis=1)

                            train_supp_dict["index"] = train_df_copy.index
                            test_supp_dict["index"] = test_df_copy.index

                            train_df_copy = pd.concat(
                                [
                                    train_df_copy,
                                    pd.DataFrame(train_supp_dict, index=train_supp_dict["index"]).drop("index", axis=1),
                                ],
                                axis=1,
                            )
                            test_df_copy = pd.concat(
                                [
                                    test_df_copy,
                                    pd.DataFrame(test_supp_dict, index=test_supp_dict["index"]).drop("index", axis=1),
                                ],
                                axis=1,
                            )

                    pruned_X_train = train_df_copy.copy()
                    pruned_X_test = test_df_copy.copy()
                    break

            old_sum = current_sum
    except UnboundLocalError as exc:
        raise ValueError("Please try different epsilon step") from exc

    return pruned_X_train, pruned_X_test
