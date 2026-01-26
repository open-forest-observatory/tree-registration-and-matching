from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from tree_registration_and_matching.register_MEE import align_plot
from tree_registration_and_matching.utils import ensure_projected_CRS


def score_approach(
    detected_tree_folder: str,
    field_tree_folder: str,
    plots_file: str,
    alignment_algorithm=align_plot,
    vis_plots: bool = False,
    vis_results: bool = True,
):
    # Read all the plot bounds
    all_plot_bounds = gpd.read_file(plots_file)

    # List all the files in both input folders
    detected_tree_files = list(detected_tree_folder.glob("*.gpkg"))
    field_tree_files = list(field_tree_folder.glob("*.gpkg"))

    # Ensure there are the same number of files
    if len(detected_tree_files) != len(field_tree_files):
        raise ValueError("Different number of files")

    # Record the shifts
    all_shifts = []

    # Iterate over files
    for detected_tree_file in detected_tree_files:
        # The field tree should have the same filename (different folder)
        field_tree_file = Path(field_tree_folder, detected_tree_file.name)

        # Read both files
        field_trees = gpd.read_file(field_tree_file)
        detected_trees = gpd.read_file(detected_tree_file)

        # Dataset name
        plot_id = detected_tree_file.stem[:4]
        plot_bounds = all_plot_bounds.query("plot_id == @plot_id")

        # Make sure the two datasets have the same CRS and it's meters-based
        field_trees = ensure_projected_CRS(field_trees)
        detected_trees.to_crs(field_trees.crs, inplace=True)
        plot_bounds.to_crs(field_trees.crs, inplace=True)

        # Visualize the three datasets if requested
        if vis_plots:
            f, ax = plt.subplots()
            plot_bounds.boundary.plot(
                ax=ax, color="k", markersize=2, label="plot bounds"
            )
            detected_trees.plot(ax=ax, c="b", markersize=2, label="detected trees")
            field_trees.plot(ax=ax, c="r", markersize=2, label="surveyed trees")
            plt.legend()
            plt.show()

        # Run aligment
        _, estimated_shift = alignment_algorithm(
            field_trees=field_trees, drone_trees=detected_trees, obs_bounds=plot_bounds
        )
        # Record shift
        all_shifts.append(estimated_shift)

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
