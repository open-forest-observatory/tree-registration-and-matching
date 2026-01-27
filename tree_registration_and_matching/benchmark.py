from pathlib import Path

import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from tree_registration_and_matching.register_MEE import align_plot
from tree_registration_and_matching.utils import ensure_projected_CRS


def score_approach(
    field_trees_folder: str,
    CHMs_or_detected_trees_folder: str,
    plots_file: str,
    alignment_algorithm=align_plot,
    CHM_approach: bool = False,
    vis_plots: bool = False,
    vis_results: bool = True,
    crop_to_plot_bounds: bool = False,
    buffer_distance: float = 0.0,
) -> np.array:
    """Assess the quality of the shift on a set of plots.

    Args:
        field_tree_folder (str):
            The folder of geospatial files representing the surveyed trees, one per plot. The files
            this folder should be named identically to those in `detected_tree_folder`.
        CHMs_or_detected_trees_folder (str):
            The folder of geospatial files representing either:
                * Canopy height models, one per plot. This data will be in a raster format (e.g. .tif)
                # Detected tree tops, one per plot. This data will be in a vector format (e.g. gpkg)
        plots_file (str):
            A path to a geopackage file with geometry representing the surveyed area. The 'plot_id'
            field represents which dataset (field trees plus CHMs or detected trees) files it
            corresponds to. Note that the plot_id field will not contain file extension.
        alignment_algorithm (function, optional):
            A function which aligns the field and detected trees. The second return argument should
            be the (x, y) shift. Defaults to align_plot.
        CHM_approach(bool, optional):
            Whether the approach acts on CHM data, rather than detected trees. Defaults to False.
        vis_plots (bool, optional):
            Should each plot be shown. Defaults to False.
        vis_results (bool, optional):
            Should the offset from true shifts be visualized. Defaults to True.
        crop_to_plot_bounds (bool, optional):
            Whether to crop the detected trees to the plot bounds area. Defaults to False.
        buffer_distance (float, optional):
            Buffer distance around the plot bounds to include when cropping. Defaults to 0.0.

    Raises:
        ValueError:
            If the detected and field tree folders do not have the same number of files with the
            same names.

    Returns:
        np.array: The error for each plot, ordered by the sorted plot IDs.
    """
    # Read all the plot bounds
    all_plot_bounds = gpd.read_file(plots_file)

    # List all the files in both input folders
    field_tree_files = sorted(field_trees_folder.glob("*"))
    CHMs_or_detected_trees_files = sorted(CHMs_or_detected_trees_folder.glob("*"))

    # Ensure there are the same number of files
    if len(field_tree_files) != len(CHMs_or_detected_trees_files):
        raise ValueError("Different number of files")

    if set([f.stem for f in field_tree_files]) != set(
        [f.stem for f in CHMs_or_detected_trees_files]
    ):
        raise ValueError("Different filenames")

    # Record the shifts
    all_shifts = []

    # Iterate over files
    for field_tree_file in field_tree_files:
        # TODO this should be more flexible for file extension
        CHMs_or_detected_trees_file = Path(
            CHMs_or_detected_trees_folder, field_tree_file.name
        ).with_suffix(".tif" if CHM_approach else ".gpkg")

        # Read the field trees and ensure that a projected CRS is used
        field_trees = gpd.read_file(field_tree_file)
        field_trees = ensure_projected_CRS(field_trees)

        # Read the plot name and extract the plot bounds for this dataset
        # TODO, make optional
        plot_id = field_tree_file.stem[:4]
        plot_bounds = all_plot_bounds.query("plot_id == @plot_id")
        plot_bounds.to_crs(field_trees.crs, inplace=True)

        if CHM_approach:
            content_to_register_to = rio.open(CHMs_or_detected_trees_file)
            # TODO add a raster-specific visualization approach
        else:
            # Read the detected trees and convert to the same CRS as the field trees
            content_to_register_to = gpd.read_file(CHMs_or_detected_trees_file)
            content_to_register_to.to_crs(field_trees.crs, inplace=True)

            # Crop to plot bounds if requested
            if crop_to_plot_bounds:
                buffered_bounds = plot_bounds.copy()
                buffered_bounds.geometry = buffered_bounds.buffer(buffer_distance)
                content_to_register_to = gpd.clip(
                    content_to_register_to, buffered_bounds
                )

            # Visualize the three datasets if requested
            if vis_plots:
                f, ax = plt.subplots()
                plot_bounds.boundary.plot(
                    ax=ax, color="k", markersize=2, label="plot bounds"
                )
                content_to_register_to.plot(
                    ax=ax, c="b", markersize=2, label="detected trees"
                )
                field_trees.plot(ax=ax, c="r", markersize=2, label="surveyed trees")
                plt.title(f"Visualization for {plot_id}")
                plt.legend()
                plt.show()

        # Run aligment
        _, estimated_shift = alignment_algorithm(
            field_trees,
            content_to_register_to,
            plot_bounds,
        )
        # Record shift
        all_shifts.append(estimated_shift)

        if CHM_approach:
            # In this case, content_to_register_to is an open file handler for a raster. Close it to
            # avoid a memory leak.
            content_to_register_to.close()

    all_shifts = np.array(all_shifts)
    if vis_results:
        # Plot the x,y errors
        plt.title("Scatter plot of the errors")
        plt.scatter(all_shifts[:, 0], all_shifts[:, 1])
        plt.xlim([-12, 12])
        plt.ylim([-12, 12])
        plt.show()

        # Plot the distribution of error magnitudes
        magnitudes = np.linalg.norm(all_shifts, axis=1)
        plt.title("Magnitudes of errors")
        plt.hist(magnitudes)

    return all_shifts
