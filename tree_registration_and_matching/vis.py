from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show


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
    if not isinstance(raster, rio.DatasetReader):
        raster = rio.open(raster)

    is_CHM = raster.count == 1

    if is_CHM:
        label = "CHM height (m)"
    else:
        label = None

    if ax is None:
        _, ax = plt.subplots()
    f = ax.get_figure()

    # Show the raster
    ret = show(raster, ax=ax, adjust=False)

    if is_CHM and add_colorbar:
        im = ret.get_images()[0]
        f.colorbar(im, ax=ax, label=label)

    # Show the plot bounds, if provided
    if plot_bounds is not None:
        plot_bounds.to_crs(raster.CRS, inplace=True)
        plot_bounds.plot(
            ax=ax, facecolor="none", edgecolor="cyan", linewidth=3, label="Plot bounds"
        )
        # Crop the view to the extent of the plot bounds
        minx, miny, maxx, maxy = plot_bounds.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # Show the tree points, sized by height
    tree_points.to_crs(raster.CRS, inplace=True)
    tree_points.plot(
        ax=ax,
        markersize=tree_points[height_column] * height_plotting_scale,
        c=point_color,
    )

    if title is not None:
        ax.set_title(title)

    return ax
