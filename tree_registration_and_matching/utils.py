import geopandas as gpd
import numpy as np
import pyproj
import shapely

from tree_registration_and_matching.constants import LAT_LON_CRS


def get_projected_CRS(lat, lon, assume_western_hem=True):
    if assume_western_hem and lon > 0:
        lon = -lon
    # https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
    epgs_code = 32700 - round((45 + lat) / 90) * 100 + round((183 + lon) / 6)
    crs = pyproj.CRS.from_epsg(epgs_code)
    return crs


def ensure_projected_CRS(geodata: gpd.GeoDataFrame):
    """Returns a projected geodataframe from the provided geodataframe by converting it to
    ESPG:4326 (if not already) and determining the projected CRS from the point
    coordinates.
    Args:
        geodata (gpd.GeoDataGrame): Original geodataframe that is potentially unprojected
    Returns:
        gpd.GeoDataGrame: projected geodataframe
    """
    # If CRS is projected return immediately
    if geodata.crs.is_projected:
        return geodata

    # If CRS is geographic and not long-lat, convert it to long-lat
    if geodata.crs.is_geographic and geodata.crs != LAT_LON_CRS:
        geodata = geodata.to_crs(LAT_LON_CRS)

    # Convert geographic long-lat CRS to projected CRS
    point = geodata["geometry"].iloc[0].centroid
    geometric_crs = get_projected_CRS(lon=point.x, lat=point.y)
    return geodata.to_crs(geometric_crs)


# Taken from here:
# https://stackoverflow.com/questions/6430091/efficient-distance-calculation-between-n-points-and-a-reference-in-numpy-scipy
# This is drop-in replacement for scipy.cdist
def cdist(x, y):
    """
    Compute pair-wise distances between points in x and y.

    Parameters:
        x (ndarray): Numpy array of shape (n_samples_x, n_features).
        y (ndarray): Numpy array of shape (n_samples_y, n_features).

    Returns:
        ndarray: Numpy array of shape (n_samples_x, n_samples_y) containing
        the pair-wise distances between points in x and y.
    """
    # Reshape x and y to enable broadcasting
    x_reshaped = x[:, np.newaxis, :]  # Shape: (n_samples_x, 1, n_features)
    y_reshaped = y[np.newaxis, :, :]  # Shape: (1, n_samples_y, n_features)

    # Compute pair-wise distances using Euclidean distance formula
    pairwise_distances = np.sqrt(np.sum((x_reshaped - y_reshaped) ** 2, axis=2))

    return pairwise_distances


def is_overstory(tree_dataset: gpd.GeoDataFrame, heights_column="height"):
    """
    Compute which trees are in the overstory based on heights and locations
    https://github.com/open-forest-observatory/ofo-itd-crossmapping/blob/1c35bb20f31013527c35bc56ab7bf5ef5ab1aa72/workflow/30_evaluate-predicted-trees.R#L90

    Args:
        tree_dataset (gpd.GeoDataFrame): The trees represented as points with a height column

    Returns:
        np.array: binary array representing which trees are overstory
    """
    heights = tree_dataset[heights_column].values

    # If no trees are present, return an empty index
    if len(tree_dataset) == 0:
        return np.array([], dtype=bool)

    # Compute the difference in heights between different trees. This is the j axis minus the one on
    # the i axis
    height_diffs = heights[np.newaxis, :] - heights[:, np.newaxis]

    # Get a numpy array of coordinates
    tree_points = shapely.get_coordinates(tree_dataset.geometry)

    # Compute the distances between each tree
    dists = cdist(tree_points, tree_points)

    # Compute the threshold distance for tree based on the difference in height
    dist_threshold = height_diffs * 0.1 + 1

    # Is the tree on the i axis within the threshold distance
    is_within_threshold = dists < dist_threshold

    # Is the tree on the i axis shorter than the one on the j axis
    is_shorter = height_diffs > 0

    # Are both conditions met
    shorter_and_under_threshold = np.logical_and(is_within_threshold, is_shorter)

    # Is the tree not occluded by any other trees
    is_overstory = np.logical_not(np.any(shorter_and_under_threshold, axis=1))

    return is_overstory
