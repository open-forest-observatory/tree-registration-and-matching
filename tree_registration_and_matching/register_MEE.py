import itertools
import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import translate

from tree_registration_and_matching.eval import obj_mee_matching
from tree_registration_and_matching.utils import ensure_projected_CRS


def find_best_shift(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    objective_function: typing.Callable,
    objective_function_kwargs: dict = {},
    search_window: float = 50,
    search_increment: float = 2,
    base_shift: typing.Tuple[float] = (0, 0),
    vis: bool = False,
    return_quality_metrics: bool = False,
) -> np.array:
    """
    Compute the shift for the observed trees that minimizes the mean distance between observed trees
    and the matching drone trees.


    Args:
        field_trees (gpd.GeoDataFrame):
            Dataframe of field trees
        drone_trees (gpd.GeoDataFrame):
            Dataframe of drone trees
        obs_bounds: (gpd.GeoDataFrame):
            GeoDataFrame with a single polygon geometry representing the surveyed area corresponding
            to the field survey.
        objective_function (function):
            A function that takes the drone trees and shifted field trees and computes a score.
            Higher scores imply better alignment.
        objective_function_kwargs (dict, optional):
            Additional keyword arguments to pass to the objective function. Defaults to {}.
        search_window (float, optional):
            Distance in meters to perform grid search. Defaults to 50.
        search_increment (float, optional):
            Increment in meters for grid search. Defaults to 2.
        base_shift_x (float, optional):
            Center the grid search around shifting the x of observations this much. Defaults to 0.
        base_shift_y (float, optional):
            Center the grid search around shifting the y of observations this much. Defaults to 0.
        vis (bool, optional):
            Visualize a scatter plot of the mean closest distance to drone trees for each shift.
            Defaults to False.
        return_quality_metrics (bool, optional):
            Return the shifts that were tried and the resultant metrics

    Returns:
        np.array:
            The [x, y] shift that should be applied to the field trees to align them with the
            drone trees.
        np.array (optional):
            A list of (x, y) tuples for shifts which were tested in the coarse stage.
        np.array (optional):
            A list of resultant metric values for the tested shifts.
        float (optional):
            The metric value for the optimal shift.
    """
    # Build the shifts. Note that our eventual goal is to recover a shift for the observed trees,
    # assuming the drone trees remain fixed
    x_shifts = np.arange(
        start=base_shift[0] - search_window,
        stop=base_shift[0] + search_window,
        step=search_increment,
    )
    y_shifts = np.arange(
        start=base_shift[1] - search_window,
        stop=base_shift[1] + search_window,
        step=search_increment,
    )
    shifts = [shift for shift in (itertools.product(x_shifts, y_shifts))]

    # Iterate over the shifts and compute the objective function when applied to the shifted field
    # trees
    objective_values = []
    for shift in shifts:
        # Shift the field points and observation bounds
        shifted_field_trees = field_trees.copy()
        shifted_obs_bounds = obs_bounds.copy()

        shifted_field_trees.geometry = shifted_field_trees.translate(
            xoff=shift[0], yoff=shift[1]
        )
        shifted_obs_bounds.geometry = shifted_obs_bounds.translate(
            xoff=shift[0], yoff=shift[1]
        )

        # Compute the quality of this shift
        objective_values.append(
            objective_function(
                shifted_field_trees,
                drone_trees,
                shifted_obs_bounds,
                **objective_function_kwargs,
            )
        )

    if vis:
        # Extract the x and y components of the shifts
        x = [shift[0] for shift in shifts]
        y = [shift[1] for shift in shifts]

        # Create a scatter plot of the shifts versus the quailty of the alignment
        plt.scatter(x, y, c=objective_values)
        plt.colorbar()
        plt.show()

    # Find the shift that produced the highest score
    best_shift = shifts[np.argmax(objective_values)]

    if return_quality_metrics:
        return best_shift, shifts, objective_values, np.max(objective_values)

    return best_shift


def align_plot(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    obs_bounds: gpd.GeoDataFrame,
    height_column: str = "height",
    vis: bool = False,
    lower_threshold: float = 2.0,
    upper_threshold: float = 4.0,
) -> typing.Tuple[gpd.GeoDataFrame, typing.Tuple, float]:
    """
    Align field and drone tree by finding the shift which maximizes the number of trees which are
    matched between the two sets using a height-dependant threshold.

    Args:
        field_trees (gpd.GeoDataFrame):
            Geodataframe of the tree points surveyed in field.
        drone_trees (gpd.GeoDataFrame):
            Geodataframe of the tree points detected by drone. Assumed to be a larger spatial extent
            than the field trees.
        obs_bounds (gpd.GeoDataFrame):
            Geodataframe with one row representing the extent surveyed
        height_column (str, optional):
            Which column in the tree dataframes contains height information. Defaults to "height".
        vis (bool, optional):
            Show the registered trees. Defaults to False.
        lower_threshold (float, optional):
            Used for computing a metric of shift quality. The lower bound in meters for the distance
            from the optimal shift for which points are included in the average. Defaults to 2.0.
        upper_threshold (float, optional):
            Used for computing a metric of shift quality. The upper bound in meters for the distance
            from the optimal shift for which points are included in the average. Defaults to 4.0.

    Returns:
        gpd.GeoDataFrame: The field trees after the optimal shift has been applied.
        tuple: The x, y shift in meters that should be applied to the field trees
        float: The quality metric
    """
    original_field_CRS = field_trees.crs
    # Transform the drone trees to a cartesian CRS if not already
    field_trees = ensure_projected_CRS(field_trees)

    # Ensure that drone trees are in the same CRS
    drone_trees.to_crs(field_trees.crs, inplace=True)
    obs_bounds.to_crs(field_trees.crs, inplace=True)

    # First compute a rough shift.
    coarse_shift, coarse_shifts, coarse_shifts_metrics, _ = find_best_shift(
        field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=obs_bounds,
        objective_function=obj_mee_matching,
        objective_function_kwargs={"height_column": height_column},
        search_increment=1,
        search_window=10,
        vis=vis,
        return_quality_metrics=True,
    )
    # Then compute the fine one. This is initialized from the coarse shift
    fine_shift, _, _, optimal_quality = find_best_shift(
        field_trees=field_trees,
        drone_trees=drone_trees,
        obs_bounds=obs_bounds,
        objective_function=obj_mee_matching,
        objective_function_kwargs={"height_column": height_column},
        search_window=2,
        search_increment=0.2,
        base_shift=coarse_shift,
        return_quality_metrics=True,
    )

    print(f"Rough shift: {coarse_shift}, fine shift: {fine_shift}")

    shifted_field_trees = field_trees.copy()
    # Apply the computed shift to the geometry of all field trees
    shifted_field_trees.geometry = shifted_field_trees.geometry.apply(
        lambda x: translate(x, xoff=fine_shift[0], yoff=fine_shift[1])
    )

    if vis:
        # Plot the aligned data
        _, ax = plt.subplots()

        # By chance, the size works nicely visually without any rescaling
        shifted_field_trees.plot(ax=ax, markersize=shifted_field_trees[height_column])
        drone_trees.plot(ax=ax, markersize=drone_trees[height_column])
        plt.show()

    # Convert back to the original CRS
    shifted_field_trees.to_crs(original_field_CRS, inplace=True)

    # Compute the quality of the final shift
    # Compute which tested shifts are within a ring of pre-determined min and max radius around the
    # optimal shift.

    # Compute the distance of each tested coarse shift from the final optimal shift
    diffs_from_optimal = np.linalg.norm(
        coarse_shifts - np.expand_dims(fine_shift, axis=0), axis=1
    )
    # Determine which shifts were within the treshold distnace from the optimal shift
    within_threshold = np.logical_and(
        diffs_from_optimal > lower_threshold, diffs_from_optimal < upper_threshold
    )
    within_threshold_metrics = np.array(coarse_shifts_metrics)[within_threshold]

    # Compute the difference between the optimal metric value and the average for shifts that are
    # different
    quality_metric = optimal_quality - np.mean(within_threshold_metrics)

    return shifted_field_trees, fine_shift, quality_metric


if __name__ == "__main__":
    FIELD_REF = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_field_trees.gpkg"
    DETECTED_TREES = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_000451_000446_detected.gpkg"
    PLOT_BOUNDS = "/ofo-share/repos-david/geospatial-data-registration-toolkit/data/points/0002_plot.gpkg"

    field_trees = gpd.read_file(FIELD_REF)
    detected_trees = gpd.read_file(DETECTED_TREES)
    plot_bounds = gpd.read_file(PLOT_BOUNDS)

    align_plot(field_trees, detected_trees, obs_bounds=plot_bounds)
