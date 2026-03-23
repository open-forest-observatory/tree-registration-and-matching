import argparse
from pathlib import Path

import geopandas as gpd
import rasterio as rio

from tree_registration_and_matching.register_CHM import find_best_shift
from tree_registration_and_matching.utils import ensure_projected_CRS


def register_trees_to_CHM(
    tree_points_file: Path,
    CHM_file: Path,
    output_shifted_trees: Path,
    height_col: str = "height",
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
        output_shifted_trees (Path):
            Path to save the shifted tree points. The format is inferred from the file
            extension (e.g. .gpkg, .shp, .geojson).
        height_col (str):
            Name of the column in the tree points file containing measured tree heights.
            Defaults to "height".
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

    # Run registration
    estimated_shift, metrics = find_best_shift(
        tree_points=tree_points,
        CHM=CHM,
        height_col=height_col,
        x_range=x_range,
        y_range=y_range,
        vis=vis,
    )

    # Print metadata to stdout
    print(f"Optimal shift (dx, dy): {estimated_shift}")
    print(f"Best correlation: {metrics['best_correlation']}")
    print(f"Quality ratio (second-best / best): {metrics['ratio']}")

    # Apply the shift and save
    tree_points.geometry = tree_points.geometry.translate(
        xoff=estimated_shift[0], yoff=estimated_shift[1]
    )
    # Transform back to original CRS
    tree_points.to_crs(original_CRS, inplace=True)

    # Save out
    Path(output_shifted_trees).parent.mkdir(exist_ok=True, parents=True)
    tree_points.to_file(output_shifted_trees)


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
        required=True,
        help="Path to save the shifted tree points (format inferred from extension).",
    )
    parser.add_argument(
        "--height-col",
        default="height",
        help="Name of the height column in the tree points file. Defaults to 'height'.",
    )
    parser.add_argument(
        "--x-range",
        type=float,
        nargs=3,
        default=(-10, 10, 1),
        metavar=("START", "STOP", "STEP"),
        help="Search range for x offsets as three values: start stop step. Defaults to -10 10 1.",
    )
    parser.add_argument(
        "--y-range",
        type=float,
        nargs=3,
        default=(-10, 10, 1),
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
