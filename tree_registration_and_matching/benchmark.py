from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.plot import show

from tree_registration_and_matching.register_MEE import align_plot
from tree_registration_and_matching.utils import ensure_projected_CRS


def compute_shift(
    field_trees_file,
    CHMs_or_detected_trees_file,
    plot_bounds,
    plot_id,
    alignment_algorithm,
    CHM_approach,
    crop_to_plot_bounds,
    plot_buffer_distance,
    vis_plots,
):
    """Contains the per-plot logic to support parrellization"""
    # Read the field trees and ensure that a projected CRS is used
    field_trees = gpd.read_file(field_trees_file)
    plot_bounds.to_crs(field_trees.crs, inplace=True)

    if CHM_approach:
        content_to_register_to = rio.open(CHMs_or_detected_trees_file)
        if not content_to_register_to.crs.is_projected:
            raise ValueError(
                "Raster does not have a projected CRS. This may cause errors in registration algorithms."
            )
        # Ensure the plot bounds and field trees are in the same CRS as the raster, since the
        # raster cannot be reprojected without data loss.
        # TODO all shifts reported by the registration algorithm will be relative to the CRS of
        # the raster data, which should be addressed at some point for consistency.
        plot_bounds.to_crs(content_to_register_to.crs, inplace=True)
        field_trees.to_crs(content_to_register_to.crs, inplace=True)

        # Mask CHM outside plot bounds if requested
        if crop_to_plot_bounds:
            # Create a buffered geometry from plot bounds to include areas slightly outside
            # the exact plot boundaries, based on the specified buffer_distance
            buffered_geom = plot_bounds.buffer(plot_buffer_distance).union_all()

            # Use rasterio's mask function to set all pixels outside the buffered geometry
            # to the raster's nodata value, effectively masking irrelevant terrain data.
            # Note that this may make the dimensions of the raster smaller, which is why the
            # new transform must be recorded.
            masked_data, masked_transform = mask(
                content_to_register_to,
                [buffered_geom],
                crop=True,
                nodata=content_to_register_to.nodata,
            )

            # Get the original raster's profile and update it with the new dimensions
            # and transform from the masking operation
            profile = content_to_register_to.profile
            profile.update(
                height=masked_data.shape[1],
                width=masked_data.shape[2],
                transform=masked_transform,
            )

            # Close the original raster file handler to prevent memory leaks
            content_to_register_to.close()

            # Create an in-memory raster file to hold the masked data
            memfile = MemoryFile()
            content_to_register_to = memfile.open(**profile)

            # Write the masked data to the in-memory raster
            content_to_register_to.write(masked_data)

        if vis_plots:
            _, ax = plt.subplots()
            # Read and show the first band of the raster. If we want to support approaches for
            # multiband rasters, this would need to be updated.
            cb = show(content_to_register_to.read(1), ax=ax, cmap="viridis")
            plt.colorbar(cb, ax=ax, label="Height (m)")
            # Show the plot bounds
            # The rasterio visualization is in pixel coordinates, so we must transform the
            # geospatial data of the plot bounds to match it.
            transform = content_to_register_to.transform.__invert__().to_shapely()
            plot_bounds.affine_transform(transform).plot(
                ax=ax, facecolor="none", edgecolor="cyan", linewidth=3
            )
            ax.set_title(f"CHM for plot_ID {plot_id}")
            plt.show()
    else:
        # Ensure that the field trees are in a projected (meters-based) CRS
        field_trees = ensure_projected_CRS(field_trees)
        # Read the detected trees and convert to the same CRS as the field trees
        content_to_register_to = gpd.read_file(CHMs_or_detected_trees_file)
        content_to_register_to.to_crs(field_trees.crs, inplace=True)

        # Crop to plot bounds if requested
        if crop_to_plot_bounds:
            buffered_bounds = plot_bounds.copy()
            buffered_bounds.geometry = buffered_bounds.buffer(plot_buffer_distance)
            content_to_register_to = gpd.clip(content_to_register_to, buffered_bounds)

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
    # TODO determine if this should get the plot bounds anymore since cropping already happens
    # in this driver code.
    alignment_results = alignment_algorithm(
        field_trees,
        content_to_register_to,
        plot_bounds,
    )
    # The algorithm should return the (x, y) shift as the first of potentially-many arguments
    estimated_shift = alignment_results[0]

    if CHM_approach:
        # In this case, content_to_register_to is an open file handler for a raster. Close it to
        # avoid a memory leak.
        content_to_register_to.close()

    return estimated_shift


def score_approach(
    field_trees_folder: str,
    CHMs_or_detected_trees_folder: str,
    plots_file: str,
    alignment_algorithm=align_plot,
    CHM_approach: bool = False,
    vis_plots: bool = False,
    vis_results: bool = True,
    crop_to_plot_bounds: bool = False,
    plot_buffer_distance: float = 0.0,
    n_workers=1,
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
            Whether the content to register to should be cropped to a buffered version of the plot
            bounds. The buffer distance is defined by `plot_buffer_distance`. In the case of using
            tree locations as input, all trees outside of the area are removed prior to registration.
            If using a CHM input, all regions of the CHM outside the crop are set to the datasets'
            nadata attribute.
        plot_buffer_distance (float, optional):
            Buffer distance around the plot bounds to include when cropping. Defaults to 0.0.
        n_workers (int, optional):
            Number of multiprocessing workers to use. Defaults to 1.

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
    field_trees_files = sorted(field_trees_folder.glob("*"))
    CHMs_or_detected_trees_files = sorted(CHMs_or_detected_trees_folder.glob("*"))

    # Ensure there are the same number of files
    if len(field_trees_files) != len(CHMs_or_detected_trees_files):
        raise ValueError("Different number of files")

    if set([f.stem for f in field_trees_files]) != set(
        [f.stem for f in CHMs_or_detected_trees_files]
    ):
        raise ValueError("Different filenames")

    plot_ids = []
    plot_bounds_list = []
    # Iterate over files to extract plot bounds corresponding to each
    for field_tree_file in field_trees_files:
        # Read the plot name and extract the plot bounds for this dataset
        # TODO, make optional to provide plot bounds
        plot_id = field_tree_file.stem[:4]
        plot_bounds = all_plot_bounds.query("plot_id == @plot_id")

        plot_ids.append(plot_id)
        plot_bounds_list.append(plot_bounds)

    # Compute the results with multiprocessing
    with Pool(n_workers) as p:
        # The lists below are a bit of a hack because they just repeat the same q
        n_files = len(field_trees_files)
        all_shifts = p.starmap(
            compute_shift,
            list(
                zip(
                    field_trees_files,
                    CHMs_or_detected_trees_files,
                    plot_bounds_list,
                    plot_ids,
                    [alignment_algorithm] * n_files,
                    [CHM_approach] * n_files,
                    [crop_to_plot_bounds] * n_files,
                    [plot_buffer_distance] * n_files,
                    [vis_plots] * n_files,
                )
            ),
            chunksize=1,
        )

    all_shifts = np.array(list(all_shifts))
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
