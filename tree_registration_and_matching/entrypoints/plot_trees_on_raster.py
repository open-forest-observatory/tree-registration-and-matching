import argparse
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt

from tree_registration_and_matching.vis import (
    plot_trees_on_raster as plot_trees_on_raster_vis,
)


def plot_trees_on_raster(
    raster_file: Path,
    tree_points_file: Path,
    plot_bounds_file: Optional[Path] = None,
    shift_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    height_column: str = "height",
    height_plotting_scale: float = 0.25,
    point_color: str = "red",
    title: Optional[str] = None,
    no_colorbar: bool = False,
    vis: bool = False,
):
    """Visualize tree points overlaid on a CHM or orthomosaic raster.

    TODO: Add a high-level summary of when/why this visualization is used.

    Args:
        raster_file (Path):
            Path to the raster file for either a CHM or orthomosaic.
        tree_points_file (Path):
            Path to the vector file containing tree points. Must include the column specified by
            `height_column`.
        plot_bounds_file (Path, optional):
            Path to a vector file with the boundary of the field plot to outline. If provided, the
            view is also cropped to its extent. Defaults to None.
        shift_file (Path, optional):
            Path to a shift file (.csv, one row) produced by register_trees_to_CHM. If provided, the
            shift is applied to the tree points and plot bounds before plotting. Defaults to None.
        output_file (Path, optional):
            Path to save the figure to (format inferred from extension). If not provided, the figure
            is not saved. Defaults to None.
        height_column (str, optional):
            Column in the tree points used to scale point size, so taller trees are shown as larger
            points. Defaults to "height".
        height_plotting_scale (float, optional):
            Scalar multiple applied to `height_column` to compute the marker size. Defaults to 0.25.
        point_color (str, optional):
            Color of the tree points. Defaults to "red".
        title (str, optional):
            Title for the plot. Defaults to None.
        no_colorbar (bool, optional):
            If True, do not add a colorbar when plotting a CHM. Defaults to False.
        vis (bool, optional):
            If True, show the figure interactively. Defaults to False.
    """
    if output_file is None and not vis:
        raise ValueError(
            "Neither an output file nor vis was requested, so the plot would not be saved or shown"
        )

    # Load inputs
    tree_points = gpd.read_file(tree_points_file)
    plot_bounds = (
        gpd.read_file(plot_bounds_file) if plot_bounds_file is not None else None
    )

    # Create the plot
    ax = plot_trees_on_raster_vis(
        raster=raster_file,
        tree_points=tree_points,
        plot_bounds=plot_bounds,
        shift_file=shift_file,
        height_column=height_column,
        height_plotting_scale=height_plotting_scale,
        point_color=point_color,
        title=title,
        add_colorbar=not no_colorbar,
        savefile=output_file,
    )
    f = ax.get_figure()

    if vis:
        plt.show()

    plt.close(f)


def parse_args():
    """Parse and return arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    description = (
        # TODO: Add a high-level summary of what this entrypoint is used for.
        "TODO\n\n"
        "All arguments are passed to "
        "tree_registration_and_matching.entrypoints.plot_trees_on_raster.plot_trees_on_raster "
        "which has the following documentation:\n\n" + plot_trees_on_raster.__doc__
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--raster-file",
        type=Path,
        required=True,
        help="Path to the CHM or orthomosaic raster file.",
    )
    parser.add_argument(
        "--tree-points-file",
        type=Path,
        required=True,
        help="Path to the input vector file of tree points.",
    )
    parser.add_argument(
        "--plot-bounds-file",
        type=Path,
        help="Path to a vector file of the field plot boundary. If provided, it is outlined and the view is cropped to it.",
    )
    parser.add_argument(
        "--shift-file",
        type=Path,
        help="Path to a shift summary .csv produced by register_trees_to_CHM to apply to the tree points and plot bounds.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save the figure to (format inferred from extension).",
    )
    parser.add_argument(
        "--height-column",
        default="height",
        help="Name of the height column in the tree points file, used to scale point size. Defaults to 'height'.",
    )
    parser.add_argument(
        "--height-plotting-scale",
        type=float,
        default=0.25,
        help="Scalar multiple applied to the height column to compute the marker size. Defaults to 0.25.",
    )
    parser.add_argument(
        "--point-color",
        default="red",
        help="Color of the tree points. Defaults to 'red'.",
    )
    parser.add_argument(
        "--title",
        help="Title for the plot.",
    )
    parser.add_argument(
        "--no-colorbar",
        action="store_true",
        help="Do not add a colorbar when plotting a CHM.",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show the figure interactively.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_trees_on_raster(**vars(args))
