import json
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from tree_registration_and_matching.register_MEE import align_plot
from tree_registration_and_matching.utils import ensure_projected_CRS


def score_approach(
    detected_tree_folder,
    field_tree_folder,
    plots_file,
    shifts_file,
    vis_plots=False,
    vis_results=True,
):
    # TODO what would be the most robust way to set this up
    with open(shifts_file) as infile:
        shifts = json.load(infile)

    all_plot_bounds = gpd.read_file(plots_file)

    detected_tree_files = list(detected_tree_folder.glob("*.gpkg"))
    field_tree_files = list(field_tree_folder.glob("*.gpkg"))

    if len(detected_tree_files) != len(field_tree_files):
        raise ValueError("Different number of files")

    all_shifts = []

    for detected_tree_file in detected_tree_files:
        field_tree_file = Path(field_tree_folder, detected_tree_file.name)

        field_trees = gpd.read_file(field_tree_file)
        detected_trees = gpd.read_file(detected_tree_file)

        # Dataset name
        dataset_name = detected_tree_file.stem
        shift = shifts[dataset_name][0]
        plot_id = dataset_name[:4]
        plot_bounds = all_plot_bounds.query("plot_id == @plot_id")

        # Make sure the two datasets have the same CRS and it's meters-based
        field_trees = ensure_projected_CRS(field_trees)
        detected_trees.to_crs(field_trees.crs, inplace=True)
        plot_bounds.to_crs(field_trees.crs, inplace=True)

        # Apply the shift because it was recorded relative to the un-shifted field trees
        plot_bounds.geometry = plot_bounds.translate(xoff=shift[0], yoff=shift[1])

        if vis_plots:
            f, ax = plt.subplots()
            plot_bounds.boundary.plot(
                ax=ax, color="k", markersize=2, label="plot bounds"
            )
            detected_trees.plot(ax=ax, c="b", markersize=2, label="detected trees")
            field_trees.plot(ax=ax, c="r", markersize=2, label="surveyed trees")
            plt.legend()
            plt.show()

        _, estimated_shift = align_plot(
            field_trees=field_trees, drone_trees=detected_trees, obs_bounds=plot_bounds
        )
        all_shifts.append(estimated_shift)

    all_shifts = np.array(all_shifts)

    if vis_results:
        plt.scatter(all_shifts[:, 0], all_shifts[:, 1])
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.show()
