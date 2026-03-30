import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from tree_registration_and_matching.register_CHM import find_best_shift
from tree_registration_and_matching.utils import ensure_projected_CRS

MIN_TREE_HEIGHT = 5.0
SHIFT_RANGE = (-10, 10, 1)


def cleanup_field_trees(
    ground_reference_trees: gpd.GeoDataFrame,
    min_height: Optional[float] = None,
    height_col: str = "height",
) -> gpd.GeoDataFrame:
    """
    Perform a variety of operations to standardize the data for matching
    * Remove dead trees
    * Ensure height is present, estimating allometrically from the "dbh" column if needed
    * Remove trees shorter than a cutoff, if requested

    Args:
        ground_reference_trees (gpd.GeoDataFrame):
            must have the 'dbh' and 'live_dead' columns and heights represented by the column noted in height_col
        min_height (Optional[float], optional):
            If provided, the data will be filtered to only trees with a height greater than this.
            Defaults to None.
        height_col (str, optional):
            Which column to treat as the height. Defaults to "height".

    Returns:
        gpd.GeoDataFrame: The trees with the height_col completely filled and short trees optioanlly removed
    """
    # Remove any trees marked explicitly as dead
    ground_reference_trees = ground_reference_trees[
        ground_reference_trees.live_dead != "D"
    ]

    # First replace any missing height values with pre-computed allometric values
    nan_height = ground_reference_trees[height_col].isna()
    ground_reference_trees[nan_height][height_col] = ground_reference_trees[
        nan_height
    ].height_allometric

    # For any remaining missing height values that have DBH, use an allometric equation to compute
    # the height
    nan_height = ground_reference_trees[height_col].isna()
    # These parameters were fit on paired height, DBH data from this dataset.
    allometric_height_func = lambda x: 1.3 + np.exp(
        -0.3136489123372108 + 0.84623571 * np.log(x)
    )
    # Compute the allometric height and assign it
    allometric_height = allometric_height_func(
        ground_reference_trees[nan_height].dbh.to_numpy()
    )
    ground_reference_trees.loc[nan_height, height_col] = allometric_height

    # Filter out any trees that still don't have height
    ground_reference_trees = ground_reference_trees[
        ~ground_reference_trees[height_col].isna()
    ]

    # Remove short trees if requested
    if min_height is not None:
        ground_reference_trees = ground_reference_trees[
            ground_reference_trees[height_col] > min_height
        ]

    return ground_reference_trees


def register_trees_to_CHM(
    tree_points_file: Path,
    CHM_file: Path,
    output_shifted_trees: Optional[Path] = None,
    output_shift_summary: Optional[Path] = None,
    height_col: str = "height",
    min_tree_height: Optional[float] = MIN_TREE_HEIGHT,
    x_range: tuple = (-10, 10, 1),
    y_range: tuple = (-10, 10, 1),
    vis: bool = False,
):
    """Shift field-measured tree points to best align with a canopy height model (CHM).

    Searches over a grid of (dx, dy) translations, selecting the shift that maximizes
    the Pearson correlation between field-measured tree heights and CHM-sampled heights
    at the shifted locations. The shifted trees are saved to the output file.

    Args:
        tree_points_file (Path):
            Path to the input vector file containing field-measured tree points. Must
            include a column with tree heights specified by `height_col`.
        CHM_file (Path):
            Path to the canopy height model raster file.
        output_shifted_trees (Path, optional):
            Path to save the shifted tree points. If not provided, the shifted points will not be saved.
            Defaults to None.
        output_shift_summary (Path, optional):
            Path to save the summary of the shift. The file path should have a csv extension. If not provided,
            the summary will not be provided. Defaults to None.
        height_col (str):
            Name of the column in the tree points file containing measured tree heights.
            Defaults to "height".
        min_tree_height (float, optional):
            The minimum height to use for registration. All trees are retained in the shifted output.
            Defaults to 5.0.
        x_range (tuple):
            Search range for x (easting) offsets as three values: start, stop, step.
            Defaults to (-10, 10, 1).
        y_range (tuple):
            Search range for y (northing) offsets as three values: start, stop, step.
            Defaults to (-10, 10, 1).
        vis (bool):
            If True, show interactive visualizations of the correlation surface.
            Defaults to False.
    """
    # Load inputs
    tree_points = gpd.read_file(tree_points_file)
    CHM = rio.open(CHM_file)

    # Record the CRS to transform back to at the end
    original_CRS = tree_points.crs
    # Ensure a projected CRS is used for registration
    tree_points = ensure_projected_CRS(tree_points)
    # Record the CRS that the shift should be interpreted in
    shift_CRS = tree_points.crs

    # Cleanup tree points, ensuring all have a height column
    cleaned_tree_points = cleanup_field_trees(tree_points, min_height=min_tree_height)

    # Run registration
    estimated_shift, metrics = find_best_shift(
        tree_points=cleaned_tree_points,
        CHM=CHM,
        height_col=height_col,
        x_range=x_range,
        y_range=y_range,
        vis=vis,
    )

    if output_shifted_trees is not None:
        # Apply the shift and save
        tree_points.geometry = tree_points.geometry.translate(
            xoff=estimated_shift[0], yoff=estimated_shift[1]
        )
        # Transform back to original CRS
        tree_points.to_crs(original_CRS, inplace=True)

        # Save out
        Path(output_shifted_trees).parent.mkdir(exist_ok=True, parents=True)
        tree_points.to_file(output_shifted_trees)

    if output_shift_summary:
        # Store the shift and associated metadata
        summary_df = pd.DataFrame(
            [
                {
                    "estimated_shift_x": estimated_shift[0],
                    "estimated_shift_y": estimated_shift[1],
                    "shift_CRS": shift_CRS,
                    "optimal_correlation": metrics["best_correlation"],
                    "ratio_quality_metric": metrics["ratio"],
                    "n_shift_trees": len(cleaned_tree_points),
                    "CHM_file": CHM_file,
                }
            ]
        )

        # Save out the summary
        Path(output_shift_summary).parent.mkdir(exist_ok=True, parents=True)
        summary_df.to_csv(output_shift_summary, index=False)


def parse_args():
    """Parse and return arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    description = (
        "Register field-measured tree points to a canopy height model (CHM) by "
        "finding the (dx, dy) translation that maximizes the Pearson correlation "
        "between field-measured and CHM-sampled tree heights. The shifted points "
        "are saved to the output file. Metadata (best shift, correlation, quality "
        "ratio) is printed to stdout.\n\n"
        "All arguments are passed to "
        "tree_registration_and_matching.entrypoints.register_trees_to_CHM.register_trees_to_CHM "
        "which has the following documentation:\n\n" + register_trees_to_CHM.__doc__
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--tree-points-file",
        type=Path,
        required=True,
        help="Path to the input vector file of field-measured tree points.",
    )
    parser.add_argument(
        "--CHM-file",
        type=Path,
        required=True,
        help="Path to the canopy height model raster file.",
    )
    parser.add_argument(
        "--output-shifted-trees",
        type=Path,
        help="Path to save the shifted tree points (format inferred from extension).",
    )
    parser.add_argument(
        "--output-shift-summary",
        type=Path,
        help="Path to save the summary of the shift. The file should have a .csv extension.",
    )
    parser.add_argument(
        "--height-col",
        default="height",
        help="Name of the height column in the tree points file. Defaults to 'height'.",
    )
    parser.add_argument(
        "--min-tree-height",
        help="Only use trees above this height for registration.",
        default=MIN_TREE_HEIGHT,
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=3,
        default=SHIFT_RANGE,
        metavar=("START", "STOP", "STEP"),
        help="Search range for x offsets as three values: start stop step. Defaults to -10 10 1.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=3,
        default=SHIFT_RANGE,
        metavar=("START", "STOP", "STEP"),
        help="Search range for y offsets as three values: start stop step. Defaults to -10 10 1.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show interactive visualizations of the correlation surface.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register_trees_to_CHM(**vars(args))
