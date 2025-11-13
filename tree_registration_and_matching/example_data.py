import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


def drop_understory_trees(
    df: pd.DataFrame, height_threshold: float = 10.0
) -> pd.DataFrame:
    """
    Remove understory trees (trees below a height threshold) from a tree dataframe.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'x', 'y', and 'z' (height)
        height_threshold (float): Height threshold in meters. Trees below this are removed.

    Returns:
        pd.DataFrame: Filtered dataframe with understory trees removed.
    """
    if len(df) == 0:
        return df
    return df[df["z"] >= height_threshold].copy()


def simulate_tree_maps(
    trees_per_ha: float = 250,
    trees_per_clust: float = 5,
    cluster_radius: float = 25,
    pred_extent: float = 300,
    obs_extent: float = 100,
    horiz_jitter: float = 1,
    vert_jitter: float = 5,
    false_pos: float = 0.25,
    false_neg: float = 0.25,
    drop_observed_understory: bool = True,
    height_bias: float = 0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    random_seed: int = None,
) -> dict:
    """
    Create simulated "predicted" and "observed" tree maps with customizable clustering,
    noise, and error parameters.

    Generate random, customizably clustered points in x-y space over an area wider than
    would reasonably have a field stem map (to represent drone-based tree predictions).
    The x-y coordinates are interpreted as meters.

    Parameters:
        trees_per_ha (float): Tree density in trees per hectare. Default: 250
        trees_per_clust (float): Mean number of trees per cluster. Default: 5
        cluster_radius (float): Radius of clustering in meters. Default: 25
        pred_extent (float): Width/height of predicted tree map extent in meters. Default: 300
        obs_extent (float): Width/height of observed tree map extent in meters. Default: 100
        horiz_jitter (float): Max horizontal jitter (±) added to predicted coordinates. Default: 1
        vert_jitter (float): Max vertical jitter (±) added to predicted heights. Default: 5
        false_pos (float): Fraction of observed trees to randomly remove (false negatives). Default: 0.25
        false_neg (float): Fraction of predicted trees to randomly remove (false positives). Default: 0.25
        drop_observed_understory (bool): If True, remove understory trees from observed map. Default: True
        height_bias (float): Systematic bias to add to predicted tree heights. Default: 0
        shift_x (float): Horizontal shift applied to observed map in meters. Default: 0.0
        shift_y (float): Vertical shift applied to observed map in meters. Default: 0.0
        random_seed (int): Random seed for reproducibility. Default: None

    Returns:
        dict: Dictionary with keys:
            - 'pred': DataFrame of predicted trees with columns 'x', 'y', 'z'
            - 'obs': DataFrame of observed trees with columns 'x', 'y', 'z'
            - 'obs_bounds': GeoDataFrame with bounding box geometry of observed trees
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert trees/ha to trees/m^2
    cluster_dens = trees_per_ha / trees_per_clust / 10000

    # Generate clustered points using Matérn Cluster Process
    # Calculate number of parent points
    window_area = pred_extent**2
    parent_count = np.random.poisson(cluster_dens * window_area)

    # Generate parent cluster centers
    parent_x = np.random.uniform(-pred_extent / 2, pred_extent / 2, parent_count)
    parent_y = np.random.uniform(-pred_extent / 2, pred_extent / 2, parent_count)

    # Generate offspring around each parent
    points = []
    for px, py in zip(parent_x, parent_y):
        n_offspring = np.random.poisson(trees_per_clust)
        if n_offspring > 0:
            # Generate offspring in a circle around parent
            angles = np.random.uniform(0, 2 * np.pi, n_offspring)
            radii = np.random.uniform(0, cluster_radius, n_offspring)
            offspring_x = px + radii * np.cos(angles)
            offspring_y = py + radii * np.sin(angles)

            # Only keep offspring within window
            mask = (
                (offspring_x >= -pred_extent / 2)
                & (offspring_x <= pred_extent / 2)
                & (offspring_y >= -pred_extent / 2)
                & (offspring_y <= pred_extent / 2)
            )
            points.extend(list(zip(offspring_x[mask], offspring_y[mask])))

    if len(points) == 0:
        # Return empty dataframes if no points generated
        return {
            "pred": pd.DataFrame(columns=["x", "y", "z"]),
            "obs": pd.DataFrame(columns=["x", "y", "z"]),
            "obs_bounds": gpd.GeoDataFrame(),
        }

    # Create predicted trees dataframe
    pred = pd.DataFrame(points, columns=["x", "y"])

    # Assign heights (5-50m, skewed toward lower end using normal distribution)
    heights = np.random.normal(15, 20, 10000)
    heights = heights[(heights >= 5) & (heights <= 50)]
    pred["z"] = np.random.choice(heights, size=len(pred), replace=True)

    # Sort by decreasing height (so smaller trees plot on top)
    pred = pred.sort_values("z", ascending=False).reset_index(drop=True)

    # Create observed trees as spatial subset of predicted
    obs = pred[
        (pred["x"] > -obs_extent / 2)
        & (pred["x"] < obs_extent / 2)
        & (pred["y"] > -obs_extent / 2)
        & (pred["y"] < obs_extent / 2)
    ].copy()

    # Randomly remove fraction of trees (simulate false positives/negatives)
    pred = pred.sample(frac=(1 - false_neg), replace=False).reset_index(drop=True)
    obs = obs.sample(frac=(1 - false_pos), replace=False).reset_index(drop=True)

    # Add noise to predicted x-y coordinates
    pred["x"] = pred["x"] + np.random.uniform(-horiz_jitter, horiz_jitter, len(pred))
    pred["y"] = pred["y"] + np.random.uniform(-horiz_jitter, horiz_jitter, len(pred))

    # Remove understory trees from predicted dataset
    pred = drop_understory_trees(pred)

    # Add noise and bias to predicted heights
    pred["z"] = (
        pred["z"]
        + np.random.uniform(-vert_jitter, vert_jitter, len(pred))
        + height_bias
    )

    # If no observed trees, return early
    if len(obs) == 0:
        return {"pred": pred, "obs": obs}

    # Optionally remove understory trees from observed dataset
    if drop_observed_understory:
        obs = drop_understory_trees(obs)

    # Apply spatial shift to observed map
    obs["x"] = obs["x"] + shift_x
    obs["y"] = obs["y"] + shift_y

    # Compute bounds of observed tree map
    obs_sf = gpd.GeoDataFrame(obs, geometry=gpd.points_from_xy(obs["x"], obs["y"]))
    obs_bbox = obs_sf.total_bounds  # [minx, miny, maxx, maxy]
    obs_bounds = gpd.GeoDataFrame(
        geometry=[box(obs_bbox[0], obs_bbox[1], obs_bbox[2], obs_bbox[3])], crs=None
    )

    return {"pred": pred, "obs": obs, "obs_bounds": obs_bounds}
