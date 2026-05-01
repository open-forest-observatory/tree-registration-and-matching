import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from tree_registration_and_matching.register_CHM import find_best_shift
from tree_registration_and_matching.utils import (
    ensure_height_is_present,
    ensure_projected_CRS,
)

MIN_TREE_HEIGHT = 5.0

COARSE_RANGE = (-10, 10, 1)
FINE_RANGE = (-2, 2, 0.2)


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

    # Make sure every tree has a height value, dropping any for which height cannot be imputed
    ground_reference_trees = ensure_height_is_present(
        ground_reference_trees, height_col=height_col
    )

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
    coarse_range: tuple = COARSE_RANGE,
    fine_range: tuple = FINE_RANGE,
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
        coarse_range (tuple):
            Search range for the initial coarse offsets as three values: start, stop, step.
            Defaults to (-10, 10, 1).
        fine_range (tuple):
            Search range for the secondary fine shift offsets as three values: start, stop, step. This shift is
            centered at the optimal value identified by the coarse shift. Defaults to (-2, 2, 0.2).
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

    # Run coarse registration
    coarse_shift, coarse_metrics = find_best_shift(
        tree_points=cleaned_tree_points,
        CHM=CHM,
        height_col=height_col,
        x_range=coarse_range,
        y_range=coarse_range,
        vis=vis,
    )

    # Check if the first step failed
    if np.all(np.isfinite(coarse_shift)):
        # Run fine registration
        # The range start and stops are shifted by the amount identified in the coarse shift
        fine_shift, fine_metrics = find_best_shift(
            tree_points=cleaned_tree_points,
            CHM=CHM,
            height_col=height_col,
            x_range=(
                fine_range[0] + coarse_shift[0],
                fine_range[1] + coarse_shift[0],
                fine_range[2],
            ),
            y_range=(
                fine_range[0] + coarse_shift[1],
                fine_range[1] + coarse_shift[1],
                fine_range[2],
            ),
            vis=vis,
        )
    else:
        # Default the values that are reported in the final metrics in the case where coarse
        # registration failed.
        fine_shift = (np.nan, np.nan)
        fine_metrics = {"best_correlation": np.nan, "average_error_frac": np.nan}

    if output_shifted_trees is not None:
        # Apply the shift and save
        tree_points.geometry = tree_points.geometry.translate(
            xoff=fine_shift[0], yoff=fine_shift[1]
        )
        # Transform back to original CRS
        tree_points.to_crs(original_CRS, inplace=True)

        # Save out
        Path(output_shifted_trees).parent.mkdir(exist_ok=True, parents=True)
        tree_points.to_file(output_shifted_trees)

    if output_shift_summary:
        # Store the shift and associated metadata
        # Note that the optimal correlation represents the best value found in the fine shift,
        # which corresponds to the reported shift. However, the ratio test is performed on the
        # coarse grid, because this covers a larger spatial areas which is better at determining
        # the quality of the shift compared to other options.
        summary_df = pd.DataFrame(
            [
                {
                    "estimated_shift_x": fine_shift[0],
                    "estimated_shift_y": fine_shift[1],
                    "shift_CRS": shift_CRS,
                    "optimal_correlation": fine_metrics["best_correlation"],
                    "ratio_quality_metric": coarse_metrics["ratio"],
                    "n_shift_trees": len(cleaned_tree_points),
                    "CHM_file": CHM_file,
                    "average_error_frac": fine_metrics["average_error_frac"],
                }
            ]
        )

        # Save out the summary
        Path(output_shift_summary).parent.mkdir(exist_ok=True, parents=True)
        summary_df.to_csv(output_shift_summary, index=False, na_rep="NaN")


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
        "--coarse-range",
        type=float,
        nargs=3,
        default=COARSE_RANGE,
        metavar=("START", "STOP", "STEP"),
        help="Search range for the initial coarse offsets as three values: start stop step. Defaults to -10 10 1.",
    )
    parser.add_argument(
        "--fine-range",
        type=float,
        nargs=3,
        default=FINE_RANGE,
        metavar=("START", "STOP", "STEP"),
        help="Search range for the secondary fine shift offsets as three values: start stop step. Defaults to -2 2 0.2.",
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
