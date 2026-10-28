import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from tree_registration_and_matching.eval import (
    compare_field_drone_attributes,
    obj_mee_matching,
)
from tree_registration_and_matching.utils import ensure_projected_CRS
from tree_registration_and_matching.vis import vis_tree_matches

DEFAULT_MIN_HEIGHT = 10.0
DEFAULT_EDGE_BUFFER = 5.0
DEFAULT_HEIGHT_COLUMN = "height"


def shift_field_products(
    shift_file: Path | str,
    field_trees: gpd.GeoDataFrame,
    field_plot_bounds: gpd.GeoDataFrame,
):
    """
    Read a shift from a file and apply it to both the field trees and plot bounds. The CRS is
    temporarily transformed but the returned data has the original CRS.

    Args:
        shift_file (Path | str): Path to a .csv file with one row and (at minimum) estimated_shift_x, estimated_shift_y, shift_CRS columns
        field_trees (gpd.GeoDataFrame): the unshifted field trees
        field_plot_bounds (gpd.GeoDataFrame): the unshifted field plot bounds

    Raises:
        ValueError: if the shift file does not have one data row

    Returns:
        gdf.GeoDataFrame: the shifted field trees
        gdf.GeoDataFrame: the shifted field plot bounds
    """
    # Assume the CRS is the same between both trees and plot bounds
    initial_crs = field_trees.crs

    shift = pd.read_csv(shift_file)

    if len(shift) != 1:
        raise ValueError(
            f"The shift file should contain 1 line but instead contains {len(shift)}"
        )

    estimated_shift_x = shift.estimated_shift_x.iloc[0]
    estimated_shift_y = shift.estimated_shift_y.iloc[0]
    shift_crs = shift.shift_CRS.iloc[0]

    # Ensure that the products are in the CRS the shift was computed relative to
    field_trees = field_trees.to_crs(shift_crs)
    field_plot_bounds = field_plot_bounds.to_crs(shift_crs)

    # Perform the translation
    field_trees.geometry = field_trees.translate(
        xoff=estimated_shift_x, yoff=estimated_shift_y
    )
    field_plot_bounds.geometry = field_plot_bounds.translate(
        xoff=estimated_shift_x, yoff=estimated_shift_y
    )

    # Return the data in the original crs
    return field_trees.to_crs(initial_crs), field_plot_bounds.to_crs(initial_crs)


def score_detected_trees(
    field_trees_file: Path | str,
    drone_trees_file: Path | str,
    field_plot_bounds_file: Path | str,
    shift_file: Path | str | None = None,
    output_file: Path | str | None = None,
    min_height: float = DEFAULT_MIN_HEIGHT,
    edge_buffer: float = DEFAULT_EDGE_BUFFER,
    height_column: str = DEFAULT_HEIGHT_COLUMN,
    vis: bool = False,
    vis_file: Path = None,
    matches_file: Path = None,
    field_attribute_to_score: str = None,
    drone_attribute_to_score: str = None,
    categorical_scoring_attribute: bool | None = None,
    scoring_metrics_savefile: Path | None = None,
    vis_attribute_scoring: bool = False,
    vis_attribute_scoring_savefile: Path | None = None,
    predicted_attribute_savefile: Path | None = None,
) -> dict:
    """Compute the F1 score for how many field trees were matched by drone-detected trees.
    Adapted from: https://github.com/open-forest-observatory/ofo-r/blob/3e3d138ffd99539affb7158979d06fc535bc1066/R/tree-map-alignment.R#L138

    Args:
        field_trees_file (Path | str): Path to the field surveyed trees, represented as points. Must have a column for tree height.
        drone_trees_file (Path | str): Path to the drone detected tree tops, represented as points. Must have a column for tree height.
        field_plot_bounds_file (Path | str): Path to the region that was surveyed, represented as a single polygon.
        output_file (Path | str | None, optional): Path to write the metrics to as JSON. If None, nothing is written. Defaults to None.
        min_height (float, optional): Minimum height of field trees to evaluate. Defaults to 10.
        edge_buffer (float, optional): Only score trees this distance from the boundary of the survey region. Defaults to 5.
        height_column (str, optional): What column represents the tree heights. Defaults to "height".
        vis (bool, optional): Should the matching be shown. Defaults to False.
        vis_file (Path, optional): Save the match visualizations to this file. Defaults to None.
        matches_file (Path, optional): Save out a geofile of lines representing the matches between field and drone trees. Defaults to None.
        field_attribute_to_score (str, optional): The column in the field data to use as the prediction target. Defaults to None.
        drone_attribute_to_score (str, optional): The column in the drone data to use as the prediction. Will be evaluated against the field_attribute_to_score column of the field data. Defaults to None.
        categorical_scoring_attribute (bool | None, optional): Is the attribute categorical? Defaults to None.
        scoring_metrics_savefile (Path | None, optional): Path to save the scoring metrics to in .json format. Defaults to None.
        vis_attribute_scoring (bool, optional): Should the scoring results be visualized to the screen. Defaults to False.
        vis_attribute_scoring_savefile (Path | None, optional): Path to save the prediction scoring visualization to. Defaults to None.
        predicted_attribute_savefile (Path, optional) Path to save a .csv file with one row for every field tree and the drone-predicted attribute if present in a matched tree. Defaults to None.

    Returns:
        dict:
            F1, precision, recall, and counts for drone/field core/matched trees
        dict (optional):
            The metrics for attribute prediction quality, which are only computed if
            field_attribute_to_score, drone_attribute_to_score, and categorical_scoring_attribute
            are provided. In the case of continious data, the metric keys are "R2", "pearson_r", and
            "spearman_r". For categorical data, the metrics are the outputs of
            sklearn.metrics.classification_report.
    """
    # Read the files
    field_trees = gpd.read_file(field_trees_file)
    drone_trees = gpd.read_file(drone_trees_file)
    field_plot_bounds = gpd.read_file(field_plot_bounds_file)

    # Reproject the field trees to a projected CRS
    field_trees = ensure_projected_CRS(field_trees)

    # Ensure everything is in the same (projected) CRS, using the field trees as the reference
    if drone_trees.crs != field_trees.crs:
        drone_trees = drone_trees.to_crs(field_trees.crs)
    if field_plot_bounds.crs != field_trees.crs:
        field_plot_bounds = field_plot_bounds.to_crs(field_trees.crs)

    # If requested, read and apply a shift to both field products
    if shift_file is not None:
        field_trees, field_plot_bounds = shift_field_products(
            shift_file, field_trees, field_plot_bounds
        )

    logging.info(
        f"Scoring {len(drone_trees)} drone trees against {len(field_trees)} field trees"
    )

    (
        _,
        detection_metrics,
        matched_field_tree_indices,
        matched_drone_tree_indices,
    ) = obj_mee_matching(
        shifted_field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=field_plot_bounds,
        min_height=min_height,
        edge_buffer=edge_buffer,
        height_column=height_column,
        return_all_metrics=True,
    )

    logging.info(f"Detection metrics: {detection_metrics}")

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as f:
            json.dump(detection_metrics, f, indent=2)

    # Slight chance this is a no-op but it's fast so it's easier to just run than check if anything
    # will be shown or visualized.
    vis_tree_matches(
        field_trees=field_trees,
        drone_trees=drone_trees,
        matched_field_trees_indices=matched_field_tree_indices,
        matched_drone_trees_indices=matched_drone_tree_indices,
        field_plot_bounds=field_plot_bounds,
        vis=vis,
        vis_file=vis_file,
        matches_savefile=matches_file,
    )

    # Determine whether attribute prediction quality should be scored
    if (
        (field_attribute_to_score is not None)
        or (drone_attribute_to_score is not None)
        or (categorical_scoring_attribute is not None)
    ):
        f_nnone = field_attribute_to_score is not None
        d_nnone = drone_attribute_to_score is not None
        c_nnone = categorical_scoring_attribute is not None

        if np.any(np.logical_not((f_nnone, d_nnone, c_nnone))):
            raise ValueError(
                "The field attribute to score, drone attribute to score, and categorical should all "
                + "be set but instead the following were provided: "
                + f"{(field_attribute_to_score, drone_attribute_to_score, categorical_scoring_attribute)}"
            )

        attribute_prediction_metrics = compare_field_drone_attributes(
            field_trees=field_trees,
            drone_trees=drone_trees,
            matched_field_trees_indices=matched_field_tree_indices,
            matched_drone_trees_indices=matched_drone_tree_indices,
            field_trees_attribute=field_attribute_to_score,
            drone_trees_attribute=drone_attribute_to_score,
            categorical=categorical_scoring_attribute,
            vis=vis_attribute_scoring,
            vis_savefile=vis_attribute_scoring_savefile,
        )

        # Whether to save out a file containing a row for every field tree, with drone predictions
        # for the corresponding class where available.
        if predicted_attribute_savefile is not None:
            # Select the field attribute, which should be included for every row
            attributes_to_save = field_trees[[field_attribute_to_score]].copy()
            # Append _field to the field data
            attributes_to_save.rename(
                columns={field_attribute_to_score: field_attribute_to_score + "_field"},
                inplace=True,
            )
            # Add the drone values to the corresponding rows as a new column. Unmatched field trees
            # get NaN. The drone values are ordered like the matched field trees, so assign the raw
            # values to avoid pandas aligning on the (different) drone tree index labels.
            drone_column = drone_attribute_to_score + "_drone"
            attributes_to_save[drone_column] = np.nan

            # Extract the values from the drone dataset
            matched_drone_attributes = drone_trees.loc[
                matched_drone_tree_indices, drone_attribute_to_score
            ].to_numpy()

            # Enter them into the corresponding rows
            attributes_to_save.loc[matched_field_tree_indices, drone_column] = (
                matched_drone_attributes
            )

            # Save out
            predicted_attribute_savefile.parent.mkdir(exist_ok=True, parents=True)
            attributes_to_save.to_csv(predicted_attribute_savefile)

        logging.info(
            f"Prediction metrics for field:{field_attribute_to_score} vs. drone:{drone_attribute_to_score}: {attribute_prediction_metrics}"
        )
        if scoring_metrics_savefile is not None:
            scoring_metrics_savefile.parent.mkdir(exist_ok=True, parents=True)
            with open(scoring_metrics_savefile, "w") as outfile:
                json.dump(attribute_prediction_metrics, outfile)

        return detection_metrics, attribute_prediction_metrics
    else:
        return detection_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the F1 score and other metrics for how many field trees were matched by drone-detected trees"
    )
    parser.add_argument(
        "field_trees_file",
        type=Path,
        help="Path to the geofile of field surveyed trees, represented as points."
        + " Note that any applicable translation should be applied to align them spatial with the drone products."
        + " OR the --shift-file argument should be provided.",
    )
    parser.add_argument(
        "drone_trees_file",
        type=Path,
        help="Path to the geofile of drone detected tree tops, represented as points.",
    )
    parser.add_argument(
        "field_plot_bounds_file",
        type=Path,
        help="Path to the geofile of region that was surveyed, represented as a single polygon."
        + " Note that any translation applied to the field trees should be applied to the plot bounds as well."
        + " OR the --shift-file argument should be provided.",
    )
    parser.add_argument(
        "--shift-file",
        type=Path,
        help="A .csv file containing (at minimum) estimated_shift_x, estimated_shift_y, shift_CRS columns. If provided, these values will be parsed and used to translate both the field trees and the field plot bounds.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to write the metrics to as JSON. If not provided, the metrics are only logged to the terminal.",
    )
    parser.add_argument(
        "--min-height",
        type=float,
        default=DEFAULT_MIN_HEIGHT,
        help=f"Minimum height of field trees to evaluate. Defaults to {DEFAULT_MIN_HEIGHT}.",
    )
    parser.add_argument(
        "--edge-buffer",
        type=float,
        default=DEFAULT_EDGE_BUFFER,
        help=f"Only score trees this distance from the boundary of the survey region. Defaults to {DEFAULT_EDGE_BUFFER}.",
    )
    parser.add_argument(
        "--height-column",
        default=DEFAULT_HEIGHT_COLUMN,
        help=f"What column in both datasets represents the tree heights. Defaults to '{DEFAULT_HEIGHT_COLUMN}'.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show the matching between field and drone trees.",
    )
    parser.add_argument(
        "--vis-file",
        type=Path,
        help="Save the matching visualization to this file if provided.",
    )
    parser.add_argument(
        "--matches-file",
        type=Path,
        help="Save out a geofile of lines representing the matches between field and drone trees.",
    )
    parser.add_argument(
        "--field-attribute-to-score",
        help="The column in the field data to use as the prediction target.",
    )
    parser.add_argument(
        "--drone-attribute-to-score",
        help="The column in the drone data to use as the prediction. Will be evaluated against the --field-attribute-to-score column of the field data.",
    )
    attribute_type_group = parser.add_mutually_exclusive_group()
    attribute_type_group.add_argument(
        "--categorical-scoring-attribute",
        help="Is the attribute to score categorical?",
        action="store_true",
        dest="categorical_scoring_attribute",
        default=None,
    )
    attribute_type_group.add_argument(
        "--continuous-scoring-attribute",
        help="Is the attribute to score continuous? Opposite of --categorical-scoring-attribute.",
        action="store_false",
        dest="categorical_scoring_attribute",
        default=None,
    )
    parser.add_argument(
        "--scoring-metrics-savefile",
        type=Path,
        help="Path to save the scoring metrics to in .json format.",
    )
    parser.add_argument(
        "--vis-attribute-scoring",
        action="store_true",
        help="Should the scoring results be visualized to the screen.",
    )
    parser.add_argument(
        "--vis-attribute-scoring-savefile",
        type=Path,
        help="Path to save the prediction scoring visualization to.",
    )
    parser.add_argument(
        "--predicted-attribute-savefile",
        type=Path,
        help="Path to save a .csv file with one row for every field tree and the drone-predicted attribute if present in a matched tree.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    score_detected_trees(**vars(args))
