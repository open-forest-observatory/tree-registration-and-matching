import geopandas as gpd
import numpy as np

from tree_registration_and_matching.register_MEE import \
    match_trees_singlestratum
from tree_registration_and_matching.utils import ensure_projected_CRS


def match_field_and_drone_trees(
    field_trees: gpd.GeoDataFrame,
    drone_trees: gpd.GeoDataFrame,
    drone_crowns: gpd.GeoDataFrame,
    field_perim: gpd.GeoDataFrame,
    field_buffer_dist: float = 10.0,
) -> gpd.GeoDataFrame:
    """
    Add information from the field points to the drone crowns, using the corresponding tree tops
    to link the two.

    Args:
        field_trees (gpd.GeoDataFrame): The field surveyed trees
        drone_trees (gpd.GeoDataFrame): The detected tree tops
        drone_crowns (gpd.GeoDataFrame): The detected crowns
        field_perim (gpd.GeoDataFrame): GeoDataFrame with a single polygon geometry representing the surveyed area corresponding
        field_buffer_dist (float, optional): Consider matching to drone trees within this distance of the field trees. Defaults to 10.0.

    Returns:
        gpd.GeoDataFrame: Drone crowns with additional attributes from the field survey
    """
    # Ensure they are all in the same projected CRS
    field_trees = ensure_projected_CRS(field_trees)
    drone_trees = drone_trees.to_crs(field_trees.crs)
    drone_crowns = drone_crowns.to_crs(field_trees.crs)
    field_perim = field_perim.to_crs(field_trees.crs)

    # Get the buffered perimiter
    perim_buff = field_perim.buffer(field_buffer_dist).geometry.values[0]

    # Consider within vs intersects or other options
    drone_trees = drone_trees[drone_trees.within(perim_buff)]
    drone_trees.index = np.arange(len(drone_trees))

    # Maybe filter some of the short trees
    # Compute the full distance matrix or at least the top n matches
    matched_field_tree_inds, matched_drone_tree_inds = match_trees_singlestratum(
        field_trees=field_trees, drone_trees=drone_trees, vis=False
    )

    # Compute field trees that were matched
    matched_field_trees = field_trees.iloc[matched_field_tree_inds]
    # Drop the geometry from the field trees since we don't want to keep it
    matched_field_trees.drop("geometry", axis=1, inplace=True)
    # Compute the "unique_ID" for matched drone trees. This is a crosswalk with the
    # "treetop_unique_ID" field in the crown polygons
    drone_tree_unique_IDs = drone_trees.iloc[
        matched_drone_tree_inds
    ].unique_ID.to_numpy()
    # These two variables, matched_field_trees and drone_tree_unique_IDs, are now ordered in the same way
    # This means corresponding rows should be paired. Effectively, we could add the
    # drone_tree_unique_ID as a column of the field trees and then merge based on that. But we don't
    # want to modify the dataframe, so it's just provided for the `merge` step.

    # Transfer the attributes to the drone trees.
    drone_crowns_with_additional_attributes = pd.merge(
        left=drone_crowns,
        right=matched_field_trees,
        left_on="treetop_unique_ID",
        right_on=drone_tree_unique_IDs,
        how="left",
        suffixes=(
            "_drone",
            "_field",
        ),  # Append these suffixes in cases of name collisions
    )

    return drone_crowns_with_additional_attributes
