from collections import OrderedDict
from pathlib import Path

import numpy as np
import shapely
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show
import matplotlib.collections as mc

from tree_registration_and_matching.utils import ensure_projected_CRS


def plot_trees_on_raster(
    raster: rio.DatasetReader | Path | str,
    tree_points: gpd.GeoDataFrame,
    plot_bounds: gpd.GeoDataFrame = None,
    height_column: str = "height",
    height_plotting_scale: float = 0.25,
    point_color: str = "red",
    title: str = None,
    ax: plt.Axes = None,
    add_colorbar: bool = True,
):
    """Visualize tree points overlaid on a CHM or orthomosaic raster.

    Args:
        raster (rio.DatasetReader, Path, str):
            An open rasterio dataset handle or path to a file for either a CHM or orthomosaic.
        tree_points (gpd.GeoDataFrame, Path, str):
            Tree point locations or a path to them.
        plot_bounds (gpd.GeoDataFrame, optional):
            Boundary of the field plot to outline. If provided, the view is also cropped to its
            extent. Defaults to None.
        height_column (str, optional): Column in `tree_points` used to scale point size, so taller
            trees are shown as larger points. Defaults to "height".
        height_plotting_scale (float, optional):
            Scalar multiple applied to `height_column` to compute the marker size. Defaults to
            0.25.
        point_color (str, optional): Color of the tree points. Defaults to "red".
        title (str, optional): Title for the plot. Defaults to None.
        ax (plt.Axes, optional): Axes to plot on. If not provided, a new figure and axes are
            created. Defaults to None.
        add_colorbar (bool):
            Whether to add a colorbar to the plot. Defaults to True.

    Returns:
        plt.Axes: The axes the data was plotted on.
    """
    opened_raster = False
    if not isinstance(raster, rio.DatasetReader):
        raster = rio.open(raster)
        opened_raster = True

    is_CHM = raster.count == 1

    if is_CHM:
        label = "CHM height (m)"
    else:
        label = None

    if ax is None:
        _, ax = plt.subplots()
    f = ax.get_figure()

    # If plot bounds are provided, compute a raster window covering just that
    # extent so only the relevant portion of the (potentially very large) raster
    # is read into memory, rather than the whole file.
    window = None
    window_transform = None
    if plot_bounds is not None:
        plot_bounds.to_crs(raster.crs, inplace=True)
        minx, miny, maxx, maxy = plot_bounds.total_bounds
        window = rio.windows.from_bounds(
            minx, miny, maxx, maxy, transform=raster.transform
        ).intersection(rio.windows.Window(0, 0, raster.width, raster.height))
        window_transform = raster.window_transform(window)

    # Show the raster, restricted to the plot-bounds window if one was computed
    if window is not None:
        if raster.count <= 2:
            data = raster.read(1, window=window, masked=True)
        else:
            try:
                # Look up the RGB band indexes from the color interpretation
                # metadata, matching what rasterio.plot.show does internally
                # for a full dataset read.
                source_colorinterp = OrderedDict(
                    zip(raster.colorinterp, raster.indexes)
                )
                rgb_indexes = [
                    source_colorinterp[ci]
                    for ci in (
                        rio.enums.ColorInterp.red,
                        rio.enums.ColorInterp.green,
                        rio.enums.ColorInterp.blue,
                    )
                ]
            except KeyError:
                rgb_indexes = (1, 2, 3)
            data = raster.read(rgb_indexes, window=window, masked=True)
        ret = show(data, transform=window_transform, ax=ax, adjust=False)
    else:
        ret = show(raster, ax=ax, adjust=False)

    if is_CHM and add_colorbar:
        im = ret.get_images()[0]
        f.colorbar(im, ax=ax, label=label)

    # Show the plot bounds outline, if provided, and crop the view to its extent
    if plot_bounds is not None:
        plot_bounds.plot(
            ax=ax, facecolor="none", edgecolor="cyan", linewidth=3, label="Plot bounds"
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # Show the tree points, sized by height
    tree_points.to_crs(raster.crs, inplace=True)
    tree_points.plot(
        ax=ax,
        markersize=tree_points[height_column] * height_plotting_scale,
        c=point_color,
    )

    if title is not None:
        ax.set_title(title)

    if opened_raster:
        raster.close()
    return ax


def vis_tree_matches(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    matched_field_trees_indices: np.ndarray,
    matched_drone_trees_indices: np.ndarray,
    field_plot_bounds: gpd.GeoDataFrame = None,
    points_scale_factor: float = 0.4,
    height_col: str = "height",
    vis: bool = False,
    vis_file: Path | None = None,
    matches_savefile: Path | None = None,
):
    """Visualize matches between field and drone trees

    Args:
        field_trees (gpd.GeoDataFrame): all field trees
        drone_trees (gpd.GeoDataFrame): all drone trees
        matched_field_trees_indices (np.ndarray): the indices (index column values) of the field trees that matched
        matched_drone_trees_indices (np.ndarray): the indices (index column values) of the drone trees that matched. In the corresponding order as the field trees.
        field_plot_bounds (gpd.GeoDataFrame, optional): The extent of the area that was surveyed. Defaults to None.
        points_scale_factor (float, optional): Scalar multiple applied to the tree height before plotting. Defaults to 0.4.
        height_col (str, optional): What column to pull height information from. Defaults to "height".
        vis (bool, optional): Should the results be shown to the screen. Defaults to False.
        vis_file (Path | None, optional): Should the results be written to this file. Defaults to None.
        matches_savefile (Path | None, optional): Should the lines visualizing the matches be written to this file. Defaults to None.
    """
    # Ensure all products are in the same projected CRS
    field_trees = ensure_projected_CRS(field_trees)
    drone_trees = drone_trees.to_crs(field_trees.crs)
    if field_plot_bounds is not None:
        field_plot_bounds = field_plot_bounds.to_crs(field_trees.crs)

    # Visualize trees form both sets
    _, ax = plt.subplots()
    field_trees.plot(
        markersize=field_trees[height_col] * points_scale_factor,
        c="#57d9d0",
        label="ground reference trees",
        ax=ax,
    )
    drone_trees.plot(
        markersize=drone_trees[height_col] * points_scale_factor,
        c="#b7484b",
        label="drone trees",
        ax=ax,
    )

    # Visualize match lines
    ordered_matched_field_tree_points = shapely.get_coordinates(
        field_trees.loc[matched_field_trees_indices].geometry
    )
    ordered_matched_drone_tree_points = shapely.get_coordinates(
        drone_trees.loc[matched_drone_trees_indices].geometry
    )
    match_lines = [
        [tuple(x), tuple(y)]
        for x, y in zip(
            ordered_matched_field_tree_points, ordered_matched_drone_tree_points
        )
    ]

    lc = mc.LineCollection(match_lines, colors="#e35eac", linewidths=2, label="matches")
    ax.add_collection(lc)

    # Visualize field plot bounds
    if field_plot_bounds is not None:
        field_plot_bounds.boundary.plot(ax=ax, color="0.5", label="Plot bounds")

    ax.set_aspect("equal")
    ax.legend()

    # Plot if requested
    if vis:
        plt.show()

    # Write to file if requested
    if vis_file is not None:
        vis_file.parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(vis_file)

    # Save out file of matches if requested. These are represented as lines connecting the matched
    # trees.
    if matches_savefile is not None:
        match_lines_gdf = gpd.GeoDataFrame(
            geometry=[shapely.LineString(ml) for ml in match_lines], crs=field_trees.crs
        )
        matches_savefile.parent.mkdir(exist_ok=True, parents=True)
        match_lines_gdf.to_file(matches_savefile)
