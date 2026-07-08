import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show


def plot_trees_on_CHM(
    CHM: rio.DatasetReader,
    tree_points: gpd.GeoDataFrame,
    plot_bounds: gpd.GeoDataFrame = None,
    height_column: str = "height",
    height_plotting_scale: float = 0.25,
    point_color: str = "red",
    title: str = None,
    ax: plt.Axes = None,
):
    """Visualize tree points overlaid on a CHM raster.

    Args:
        CHM (rio.DatasetReader): An open rasterio dataset handle for the CHM.
        tree_points (gpd.GeoDataFrame): Tree point locations, in the same CRS as the CHM.
        plot_bounds (gpd.GeoDataFrame, optional):
            Boundary of the field plot to outline, in the same CRS as the CHM. Not shown if not
            provided. Defaults to None.
        height_column (str, optional): Column in `tree_points` used to scale point size, so taller
            trees are shown as larger points. Defaults to "height".
        height_plotting_scale (float, optional):
            Scalar multiple applied to `height_column` to compute the marker size. Defaults to
            0.25.
        point_color (str, optional): Color of the tree points. Defaults to "red".
        title (str, optional): Title for the plot. Defaults to None.
        ax (plt.Axes, optional): Axes to plot on. If not provided, a new figure and axes are
            created. Defaults to None.

    Returns:
        plt.Axes: The axes the data was plotted on.
    """
    if ax is None:
        _, ax = plt.subplots()
    f = ax.get_figure()

    # Show the CHM
    ret = show(CHM, ax=ax, adjust=False)
    im = ret.get_images()[0]
    f.colorbar(im, ax=ax, label="CHM height (m)")

    # Show the plot bounds, if provided
    if plot_bounds is not None:
        plot_bounds.plot(
            ax=ax, facecolor="none", edgecolor="cyan", linewidth=3, label="Plot bounds"
        )

    # Show the tree points, sized by height
    tree_points.plot(
        ax=ax,
        markersize=tree_points[height_column] * height_plotting_scale,
        c=point_color,
    )

    if title is not None:
        ax.set_title(title)

    return ax
