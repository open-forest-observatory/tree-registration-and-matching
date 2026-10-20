import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from tree_registration_and_matching.eval import obj_mee_matching
from tree_registration_and_matching.utils import ensure_projected_CRS

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


def score_tree_detection(
    field_trees_file: Path | str,
    drone_trees_file: Path | str,
    field_plot_bounds_file: Path | str,
    shift_file: Path | str | None = None,
    output_file: Path | str | None = None,
    min_height: float = DEFAULT_MIN_HEIGHT,
    edge_buffer: float = DEFAULT_EDGE_BUFFER,
    height_column: str = DEFAULT_HEIGHT_COLUMN,
    vis: bool = False,
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

    Returns:
        dict: F1, precision, recall, and counts for drone/field core/matched trees
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

    _, metrics = obj_mee_matching(
        shifted_field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=field_plot_bounds,
        min_height=min_height,
        edge_buffer=edge_buffer,
        height_column=height_column,
        return_all_metrics=True,
        vis=vis,
    )
    logging.info(f"Metrics: {metrics}")

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    score_tree_detection(**vars(args))
