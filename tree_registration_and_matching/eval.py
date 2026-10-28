from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import shapely
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from tree_registration_and_matching.utils import cdist


def match_trees_singlestratum(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    search_height_proportion: float = 0.2,
    search_distance_fun_slope: float = 0.1,
    search_distance_fun_intercept: float = 1,
    height_col: str = "height",
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        field_trees (gpd.GeoDataFrame): _description_
        drone_trees (gpd.GeoDataFrame): _description_
        search_height_proportion (float, optional): _description_. Defaults to 0.2.
        search_distance_fun_slope (float, optional): _description_. Defaults to 0.1.
        search_distance_fun_intercept (float, optional): _description_. Defaults to 1.
        field_bounds_for_vis (Optional[gpd.GeoDataFrame], optional): _description_. Defaults to None.
        height_col (str, optional): _description_. Defaults to "height".
        vis (bool, optional): _description_. Defaults to False.
        vis_file (Path, optional): _description_. Defaults to None.
        points_scale_factor (float, optional): _description_. Defaults to 0.4.

    Returns:
        np.ndarray: indices (index column) of the matched field trees
        np.ndarray: indices (index column) of the matched drone trees
    """
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

    # Extract the index values rather than row indices
    matched_field_tree_original_indices = field_trees.index[
        matched_field_tree_inds
    ].values
    matched_drone_tree_original_indices = drone_trees.index[
        matched_drone_tree_inds
    ].values
    return matched_field_tree_original_indices, matched_drone_tree_original_indices


def obj_mee_matching(
    shifted_field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    min_height: float = 10,
    edge_buffer: float = 5,
    height_column: str = "height",
    return_all_metrics: bool = False,
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
        return_all_metrics (bool, optional): Return a dictionary of metrics as the second return.
        vis (bool, optional): Should the matching be shown. Defaults to False.
        vis_file (Path, optional): Save the match visualizations to this file. Defaults to None.
        matches_file (Path, optional): Save a geofile of lines representing matches between field and drone trees. Defaults to None.

    Returns:
        float: The F1 score for matching
        dict: (optional): f1, recall, precision, counts for drone/field core/matched
    """
    # Crop to the observation bounds
    shifted_field_trees_cropped = shifted_field_trees.clip(
        obs_bounds.geometry.values[0]
    )
    drone_trees_cropped = drone_trees.clip(obs_bounds.geometry.values[0])

    # Compute the matches between the shifted field points and the drone points
    matched_field_tree_indices, matched_drone_tree_indices = match_trees_singlestratum(
        field_trees=shifted_field_trees_cropped,
        drone_trees=drone_trees_cropped,
        height_col=height_column,
    )

    # From the tree inds, we get the number of matches. Now all that's left to do is compute which
    # fraction of those fall within the core area and how many remain

    obs_bounds_core = obs_bounds.geometry.values[0].buffer(-edge_buffer)

    # Crop to the central area to avoid edge effects
    core_field_trees = shifted_field_trees_cropped.clip(obs_bounds_core)
    core_drone_trees = drone_trees_cropped.clip(obs_bounds_core)
    # Remove the short trees
    core_field_trees = core_field_trees[core_field_trees[height_column] >= min_height]
    core_drone_trees = core_drone_trees[core_drone_trees[height_column] >= min_height]

    # Find which of the core trees were matched based on their indices
    field_core_matched = set(matched_field_tree_indices).intersection(
        set(core_field_trees.index)
    )
    drone_core_matched = set(matched_drone_tree_indices).intersection(
        set(core_drone_trees.index)
    )

    # Compute the number of trees
    n_core_field = len(core_field_trees)
    n_core_drone = len(core_drone_trees)
    n_field_matched = len(field_core_matched)
    n_drone_matched = len(drone_core_matched)

    # Compute precision and recall, using only the core trees as the denominator
    recall = n_field_matched / n_core_field if n_core_field > 0 else 0
    precision = n_drone_matched / n_core_drone if n_core_drone > 0 else 0

    # Compute the F1 score, setting to 0 if both precision and recall are 0
    P_plus_R = precision + recall
    F1 = (2 * (precision * recall) / (P_plus_R)) if (P_plus_R) > 0 else 0

    if return_all_metrics:
        # Extract the index column so the un-cropped trees can be matched
        return (
            F1,
            {
                "F1": F1,
                "precision": precision,
                "recall": recall,
                "n_core_field": n_core_field,
                "n_core_drone": n_core_drone,
                "n_field_matched": n_field_matched,
                "n_drone_matched": n_drone_matched,
            },
            matched_field_tree_indices,
            matched_drone_tree_indices,
        )

    return F1


def compare_field_drone_attributes(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    matched_field_trees_indices: np.ndarray,
    matched_drone_trees_indices: np.ndarray,
    field_trees_attribute: str,
    drone_trees_attribute: str,
    categorical: bool,
    vis: bool = False,
    vis_savefile: Path = None,
) -> dict:
    field_values_to_compare = field_trees.loc[
        matched_field_trees_indices, field_trees_attribute
    ]
    drone_values_to_compare = drone_trees.loc[
        matched_drone_trees_indices, drone_trees_attribute
    ]

    if categorical:
        field_values_to_compare = field_values_to_compare.to_list()
        drone_values_to_compare = drone_values_to_compare.to_list()

        ConfusionMatrixDisplay.from_predictions(
            field_values_to_compare, drone_values_to_compare
        )
        metrics = classification_report(
            field_values_to_compare, drone_values_to_compare, output_dict=True
        )
    else:
        metrics = {
            "R2": r2_score(field_values_to_compare, drone_values_to_compare),
            "pearson_r": float(
                pearsonr(field_values_to_compare, drone_values_to_compare).statistic
            ),
            "spearman_r": float(
                spearmanr(field_values_to_compare, drone_values_to_compare).statistic
            ),
        }

        _, ax = plt.subplots()
        ax.scatter(field_values_to_compare, drone_values_to_compare)
        # Plot 1-1 line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        # now plot both limits against eachother
        ax.plot(lims, lims, c="r", zorder=0, label="1:1 line")

        plt.xlabel(f"Field-derived values: {field_trees_attribute}")
        plt.ylabel(f"Drone-derived values: {drone_trees_attribute}")
        plt.legend()

    if vis:
        plt.show()

    if vis_savefile:
        vis_savefile.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(vis_savefile)

    plt.close()

    return metrics
