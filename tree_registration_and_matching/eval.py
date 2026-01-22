import geopandas as gpd
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import shapely

from tree_registration_and_matching.utils import cdist


def match_trees_singlestratum(
    field_trees,
    drone_trees,
    search_height_proportion=0.5,
    search_distance_fun_slope=0.1,
    search_distance_fun_intercept=1,
    height_col="height",
    vis=False,
):
    # A reimplementation of
    # https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-detection-accuracy-assessment.R#L164
    # Compute the pairwise distance matrix (dense, I don't see a way around it)
    field_tree_points_np = shapely.get_coordinates(field_trees.geometry)
    drone_tree_points_np = shapely.get_coordinates(drone_trees.geometry)

    # consider if this order should be switched
    # This looks correct, it seems like the observed trees are vertical
    distance_matrix = cdist(field_tree_points_np, drone_tree_points_np)

    # Expand so the field trees are a tall matrix and the drone trees are a wide one
    field_height = np.expand_dims(field_trees[height_col].to_numpy(), axis=1)
    drone_height = np.expand_dims(drone_trees[height_col].to_numpy(), axis=0)

    # Compute upper and lower height bounds for matches
    min_drone_height = field_height * (1 - search_height_proportion)
    max_drone_height = field_height * (1 + search_height_proportion)
    # Compute max spatial distances for valid matches
    max_dist = field_height * search_distance_fun_slope + search_distance_fun_intercept

    # Compute which matches fit the criteria using broadcasting to get a matrix representation
    above_min_height = drone_height > min_drone_height
    below_max_height = drone_height < max_drone_height
    below_max_matching_dist = distance_matrix < max_dist

    # Compute which matches fit all three criteria
    possible_pairings = np.logical_and.reduce(
        [above_min_height, below_max_height, below_max_matching_dist]
    )

    # Extract the indices of possible pairings
    possible_pairing_field_inds, possible_paring_drone_inds = np.where(
        possible_pairings
    )
    possible_pairing_inds = np.vstack(
        [possible_pairing_field_inds, possible_paring_drone_inds]
    ).T

    # Extract the distances corresponding to the valid matches
    possible_dists = distance_matrix[
        possible_pairing_field_inds, possible_paring_drone_inds
    ]

    # Sort so the paired indices are sorted, corresponding to the smallest distance pair first
    ordered_by_dist = np.argsort(possible_dists)
    possible_pairing_inds = possible_pairing_inds[ordered_by_dist]

    # Compute the most possible pairs, which is the min of num field and drone trees
    max_valid_matches = np.min(distance_matrix.shape)

    # Record the valid mathces
    matched_field_tree_inds = []
    matched_drone_tree_inds = []

    # Iterate over the indices
    for field_ind, drone_ind in possible_pairing_inds:
        # If neither the field or drone tree has already been matched, this is a valid pairing
        if (field_ind not in matched_field_tree_inds) and (
            drone_ind not in matched_drone_tree_inds
        ):
            # Add the matches to the lists
            matched_field_tree_inds.append(field_ind)
            matched_drone_tree_inds.append(drone_ind)

        # Check to see if all possible trees have been matched. Note, the length of matched field
        # and matched drone inds is the same, so we only need to check one.
        if len(matched_field_tree_inds) == max_valid_matches:
            break

    if vis:
        # Visualize matches
        f, ax = plt.subplots()
        ax.scatter(x=field_tree_points_np[:, 0], y=field_tree_points_np[:, 1], c="r")
        ax.scatter(x=drone_tree_points_np[:, 0], y=drone_tree_points_np[:, 1], c="b")

        ordered_matched_field_trees = field_tree_points_np[matched_field_tree_inds]
        ordered_matched_drone_trees = drone_tree_points_np[matched_drone_tree_inds]
        lines = [
            [tuple(x), tuple(y)]
            for x, y in zip(ordered_matched_field_trees, ordered_matched_drone_trees)
        ]

        lc = mc.LineCollection(lines, colors="k", linewidths=2)
        ax.add_collection(lc)

        plt.show()
    return matched_field_tree_inds, matched_drone_tree_inds


def obj_mee_matching(
    shifted_field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    min_height: float = 10,
    edge_buffer: float = 5,
    height_column: str = "height",
    return_prec_recall: bool = False,
) -> float:
    """
    Compute the F1 score for how many trees matched.
    Adapted from: https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-map-alignment.R#L138


    Args:
        shifted_field_trees (gpd.GeoDataFrame): The field trees with a candidate shift applied
        drone_trees (gpd.GeoDataFrame): The drone trees
        obs_bounds (gpd.GeoDataFrame): The region that was surveyed, shifted commensurately with the field trees
        min_height (float, optional): Minimum height of field trees to evaluate. Defaults to 10.
        edge_buffer (float, optional): Only score trees this distance from the boundary of the survey region. Defaults to 5.
        height_column (str, optional): What column represents the tree heights. Defaults to "height".

    Returns:
        float: The F1 score for matching
    """
    # Crop to the observation bounds
    shifted_field_trees_cropped = shifted_field_trees.clip(
        obs_bounds.geometry.values[0]
    )
    drone_trees_cropped = drone_trees.clip(obs_bounds.geometry.values[0])
    # Reset the index to integers starting at 0
    shifted_field_trees_cropped.reset_index(inplace=True, drop=True)
    drone_trees_cropped.reset_index(inplace=True, drop=True)

    # Compute the matches between the shifted field points and the drone points
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=shifted_field_trees_cropped,
        drone_trees=drone_trees_cropped,
        vis=False,
    )

    # From the tree inds, we get the number of matches. Now all that's left to do is compute which
    # fraction of those fall within the core area and how many remain

    obs_bounds_core = obs_bounds.geometry.values[0].buffer(-edge_buffer)

    # Get indices of
    core_field_trees = shifted_field_trees_cropped.within(obs_bounds_core)
    core_drone_trees = drone_trees_cropped.within(obs_bounds_core)
    # Get indices of tree that are taller than the min height
    tall_field_trees = shifted_field_trees_cropped[height_column] >= min_height
    tall_drone_trees = drone_trees_cropped[height_column] >= min_height

    # Find which of the core trees were matched based on their indices
    field_core_matched = set(matched_field_tree_inds).intersection(
        set(core_field_trees.index)
    )
    drone_core_matched = set(matched_drone_tree_inds).intersection(
        set(core_drone_trees.index)
    )

    # Compute precision and recall, using only the core trees as the denominator
    recall = (
        len(field_core_matched) / len(core_field_trees)
        if len(core_field_trees) > 0
        else 0
    )
    precision = (
        len(drone_core_matched) / len(core_drone_trees)
        if len(core_drone_trees) > 0
        else 0
    )

    # Compute the F1 score, setting to 0 if both precision and recall are 0
    f1 = (
        (2 * (precision * recall) / (precision + recall))
        if (precision + recall) > 0
        else 0
    )
    if return_prec_recall:
        return precision, recall, f1

    return f1


def obj_height_corr(
    shifted_field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    min_height: float = 10,
    edge_buffer: float = 5,
    height_column: str = "height",
    return_prec_recall: bool = False,
) -> float:
    """
    Compute the F1 score for how many trees matched.
    Adapted from: https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-map-alignment.R#L138


    Args:
        shifted_field_trees (gpd.GeoDataFrame): The field trees with a candidate shift applied
        drone_trees (gpd.GeoDataFrame): The drone trees
        obs_bounds (gpd.GeoDataFrame): The region that was surveyed, shifted commensurately with the field trees
        min_height (float, optional): Minimum height of field trees to evaluate. Defaults to 10.
        edge_buffer (float, optional): Only score trees this distance from the boundary of the survey region. Defaults to 5.
        height_column (str, optional): What column represents the tree heights. Defaults to "height".

    Returns:
        float: The F1 score for matching
    """
    # Crop to the observation bounds
    shifted_field_trees_cropped = shifted_field_trees.clip(
        obs_bounds.geometry.values[0]
    )
    drone_trees_cropped = drone_trees.clip(obs_bounds.geometry.values[0])
    # Reset the index to integers starting at 0
    shifted_field_trees_cropped.reset_index(inplace=True, drop=True)
    drone_trees_cropped.reset_index(inplace=True, drop=True)

    # Compute the matches between the shifted field points and the drone points
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=shifted_field_trees_cropped,
        drone_trees=drone_trees_cropped,
        vis=False,
    )

    matched_field_heights = shifted_field_trees_cropped.loc[
        matched_field_tree_inds, "height"
    ]
    matched_drone_heights = drone_trees_cropped.loc[matched_drone_tree_inds, "height"]
    corr = np.corrcoef(matched_field_heights, matched_drone_heights)[0, 1]
    return corr

    obs_bounds_core = obs_bounds.geometry.values[0].buffer(-edge_buffer)

    # Get mask of trees within the core area
    core_field_trees = shifted_field_trees_cropped.within(obs_bounds_core)
    core_drone_trees = drone_trees_cropped.within(obs_bounds_core)
    # Get indices of tree that are taller than the minimum height
    tall_field_trees = shifted_field_trees_cropped[height_column] >= min_height
    tall_drone_trees = drone_trees_cropped[height_column] >= min_height

    # Determine the mask of which trees are both within the core and tall enough
    valid_field_trees = core_field_trees & tall_field_trees
    valid_drone_trees = core_drone_trees & tall_drone_trees

    # Reorder such that each row is a pair
    reordered_drone_trees = shifted_field_trees_cropped.iloc[matched_field_tree_inds, :]
    reordered_field_trees = drone_trees_cropped.iloc[matched_drone_tree_inds, :]

    # Reordered validity
    reordered_field_validity = valid_field_trees.iloc[matched_field_tree_inds]
    reordered_drone_validity = valid_drone_trees.iloc[matched_drone_tree_inds]

    both_valid = reordered_field_validity & reordered_drone_validity

    #
    reordered_drone_tree_heights = reordered_drone_trees.loc[both_valid, height_column]
    reordered_field_tree_heights = reordered_field_trees.loc[both_valid, height_column]

    corr = np.corrcoef(reordered_drone_tree_heights, reordered_field_tree_heights)[0, 1]

    return corr
