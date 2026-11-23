import argparse
from pathlib import Path

from tree_registration_and_matching.vis import plot_trees_on_raster


def parse_args():
    """Parse and return arguments.

    Returns:
        argparse.Namespace: Arguments
    """
    description = (
        "Visualize tree points over a raster \n\n"
        "All arguments are passed to "
        "tree_registration_and_matching.vis.plot_trees_on_raster "
        "which has the following documentation:\n\n" + plot_trees_on_raster.__doc__
    )
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--raster",
        type=Path,
        required=True,
        help="Path to the CHM or orthomosaic raster file.",
    )
    parser.add_argument(
        "--tree-points",
        type=Path,
        required=True,
        help="Path to the input vector file of tree points.",
    )
    parser.add_argument(
        "--plot-bounds",
        type=Path,
        help="Path to a vector file of the field plot boundary. If provided, it is outlined and the view is cropped to it.",
    )
    parser.add_argument(
        "--shift-file",
        type=Path,
        help="Path to a shift summary .csv produced by register_trees_to_CHM to apply to the tree points and plot bounds.",
    )
    parser.add_argument(
        "--visualization-output-file",
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
        dest="add_colorbar",
        action="store_false",
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
    if args.visualization_output_file is None and not args.show:
        raise ValueError(
            "Neither --visualization-output-file nor --show was provided, so the plot would not be saved or shown"
        )
    plot_trees_on_raster(**vars(args))
